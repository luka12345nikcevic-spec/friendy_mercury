import math
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from .config import load_config
    from .data import YoloDetectionDataset, detection_collate_fn, load_data_yaml
    from .registry import build_model
except ImportError:
    from config import load_config
    from data import YoloDetectionDataset, detection_collate_fn, load_data_yaml
    from registry import build_model


class TrainResult(dict):
    @property
    def save_dir(self):
        return self["save_dir"]


class ResizeTransform:
    def __init__(self, size):
        self.size = int(size)

    def __call__(self, image, target):
        old_h, old_w = image.shape[-2:]
        image = F.interpolate(
            image.unsqueeze(0),
            size=(self.size, self.size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        if target["boxes"].numel():
            scale = target["boxes"].new_tensor(
                [self.size / old_w, self.size / old_h, self.size / old_w, self.size / old_h]
            )
            target = dict(target)
            target["boxes"] = target["boxes"] * scale
            target["orig_size"] = torch.tensor([self.size, self.size], dtype=torch.int64)

        return image, target


def train_from_config(config_path=None, **overrides):
    cfg = load_config(config_path, overrides=overrides or None)
    return train(cfg)


def train(cfg):
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    model_cfg = dict(cfg["model"])

    if data_cfg.get("path") is None:
        raise ValueError("Training config requires data.path pointing to a YOLO data.yaml")

    _set_seed(train_cfg.get("seed", 0), train_cfg.get("deterministic", True))
    device = _select_device(train_cfg.get("device"))
    save_dir = _make_save_dir(train_cfg.get("project", "runs/train"), train_cfg.get("name", "exp"), train_cfg.get("exist_ok", False))

    data = load_data_yaml(data_cfg["path"])
    names = data["names"]
    num_classes = model_cfg.pop("num_classes", None) or len(names)
    model_name = model_cfg.pop("name")
    model_cfg["num_classes"] = num_classes
    model_cfg = {key: value for key, value in model_cfg.items() if value is not None}

    adapter = build_model(model_name, **model_cfg).to(device)
    _freeze_layers(adapter.model, train_cfg.get("freeze"))

    train_dataset = YoloDetectionDataset(
        data_cfg["path"],
        split=data_cfg.get("train", "train"),
        transforms=ResizeTransform(train_cfg.get("imgsz", 640)),
    )
    train_dataset = _fraction_dataset(train_dataset, train_cfg.get("fraction", 1.0), train_cfg.get("seed", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg.get("batch", 16)),
        shuffle=True,
        num_workers=int(train_cfg.get("workers", 8)),
        collate_fn=detection_collate_fn,
        pin_memory=device.type == "cuda",
    )

    optimizer = build_optimizer(adapter.model, cfg["optimizer"], len(train_loader), train_cfg.get("epochs", 100))
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    start_epoch = 0
    best_loss = float("inf")
    history = []

    for epoch in range(start_epoch, int(train_cfg.get("epochs", 100))):
        epoch_stats = train_one_epoch(
            adapter=adapter,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            loss_weights=cfg.get("loss", {}).get("weights", {}),
            amp=bool(train_cfg.get("amp", True)),
        )
        epoch_stats["epoch"] = epoch + 1
        history.append(epoch_stats)

        if train_cfg.get("verbose", True):
            _print_epoch(epoch_stats, optimizer)

        if train_cfg.get("save", True):
            last_path = save_dir / "weights" / "last.pt"
            save_checkpoint(last_path, adapter, optimizer, scheduler, epoch, cfg, names)

            if epoch_stats["loss"] < best_loss:
                best_loss = epoch_stats["loss"]
                save_checkpoint(save_dir / "weights" / "best.pt", adapter, optimizer, scheduler, epoch, cfg, names)

            save_period = int(train_cfg.get("save_period", -1))
            if save_period > 0 and (epoch + 1) % save_period == 0:
                save_checkpoint(save_dir / "weights" / f"epoch{epoch + 1}.pt", adapter, optimizer, scheduler, epoch, cfg, names)

    return TrainResult({"save_dir": save_dir, "history": history, "config": cfg})


def train_one_epoch(adapter, loader, optimizer, scheduler, scaler, device, loss_weights, amp=True):
    running = {}
    total_loss = 0.0
    num_batches = 0

    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [_move_target_to_device(target, device) for target in targets]

        optimizer.zero_grad(set_to_none=True)
        _apply_warmup_momentum(optimizer, scheduler)

        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            adapter_loss, losses = adapter.training_step(images, targets)
            loss = _apply_loss_weights(adapter_loss, losses, loss_weights)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        loss_value = float(loss.detach().cpu())
        total_loss += loss_value
        num_batches += 1
        running["loss"] = running.get("loss", 0.0) + loss_value
        for key, value in losses.items():
            if torch.is_tensor(value):
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())

    return {key: value / max(num_batches, 1) for key, value in running.items()}


def build_optimizer(model, optimizer_cfg, steps_per_epoch=None, epochs=None):
    name = str(optimizer_cfg.get("name", "auto")).lower()
    lr = float(optimizer_cfg.get("lr0", 0.01))
    momentum = float(optimizer_cfg.get("momentum", 0.937))
    weight_decay = float(optimizer_cfg.get("weight_decay", 0.0005))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "auto":
        total_steps = (steps_per_epoch or 0) * (epochs or 0)
        name = "sgd" if total_steps > 10000 else "adamw"

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=bool(optimizer_cfg.get("nesterov", True)))
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    if name == "nadam":
        return torch.optim.NAdam(params, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    if name == "radam":
        return torch.optim.RAdam(params, lr=lr, betas=(momentum, 0.999), weight_decay=weight_decay)
    if name in {"rmsprop", "rms"}:
        return torch.optim.RMSprop(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    raise ValueError(f"Unsupported optimizer: {optimizer_cfg.get('name')}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    train_cfg = cfg["train"]
    opt_cfg = cfg["optimizer"]
    sched_cfg = cfg["scheduler"]
    epochs = int(train_cfg.get("epochs", 100))
    total_steps = max(epochs * steps_per_epoch, 1)
    warmup_steps = int(float(sched_cfg.get("warmup_epochs", 0.0)) * steps_per_epoch)
    lrf = float(opt_cfg.get("lrf", 0.01))
    cos_lr = bool(sched_cfg.get("cos_lr", False))

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(step / warmup_steps, 1e-6)

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if cos_lr:
            return lrf + (1.0 - lrf) * (1.0 + math.cos(math.pi * progress)) / 2.0
        return 1.0 - (1.0 - lrf) * progress

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scheduler.warmup_steps = warmup_steps
    scheduler.warmup_momentum = float(sched_cfg.get("warmup_momentum", opt_cfg.get("momentum", 0.937)))
    scheduler.base_momentum = float(opt_cfg.get("momentum", 0.937))
    return scheduler


def _apply_warmup_momentum(optimizer, scheduler):
    warmup_steps = getattr(scheduler, "warmup_steps", 0)
    if warmup_steps <= 0 or scheduler.last_epoch >= warmup_steps:
        return

    pct = max(scheduler.last_epoch, 0) / max(warmup_steps, 1)
    base_momentum = getattr(scheduler, "base_momentum", None)
    warmup_momentum = getattr(scheduler, "warmup_momentum", base_momentum)
    if base_momentum is None:
        return

    momentum = warmup_momentum + pct * (base_momentum - warmup_momentum)
    for group in optimizer.param_groups:
        if "momentum" in group:
            group["momentum"] = momentum
        elif "betas" in group:
            group["betas"] = (momentum, group["betas"][1])


def save_checkpoint(path, adapter, optimizer, scheduler, epoch, cfg, names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": adapter.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "config": cfg,
            "names": names,
            "adapter": adapter.name,
        },
        path,
    )


def _apply_loss_weights(total_loss, losses, loss_weights):
    if not loss_weights:
        return total_loss

    weighted = None
    for key, weight in loss_weights.items():
        if key in losses and torch.is_tensor(losses[key]):
            term = losses[key] * float(weight)
            weighted = term if weighted is None else weighted + term
    return total_loss if weighted is None else weighted


def _move_target_to_device(target, device):
    return {key: value.to(device, non_blocking=True) if hasattr(value, "to") else value for key, value in target.items()}


def _select_device(device):
    if device is None or device == "":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _set_seed(seed, deterministic):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def _make_save_dir(project, name, exist_ok):
    base = Path(project)
    save_dir = base / name
    if not exist_ok:
        original = save_dir
        index = 2
        while save_dir.exists():
            save_dir = Path(f"{original}{index}")
            index += 1
    (save_dir / "weights").mkdir(parents=True, exist_ok=True)
    return save_dir


def _fraction_dataset(dataset, fraction, seed):
    fraction = float(fraction)
    if fraction >= 1.0:
        return dataset
    size = max(int(len(dataset) * fraction), 1)
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:size])


def _freeze_layers(model, freeze):
    if freeze is None or freeze is False:
        return
    parameters = list(model.parameters())
    if isinstance(freeze, int):
        for param in parameters[:freeze]:
            param.requires_grad = False
        return
    if isinstance(freeze, (list, tuple, set)):
        for index in freeze:
            if 0 <= int(index) < len(parameters):
                parameters[int(index)].requires_grad = False
        return
    if freeze is True:
        for param in parameters:
            param.requires_grad = False


def _print_epoch(stats, optimizer):
    lr = optimizer.param_groups[0]["lr"]
    details = " ".join(f"{key}={value:.4f}" for key, value in stats.items() if isinstance(value, float))
    print(f"epoch={stats['epoch']} lr={lr:.6g} {details}")
