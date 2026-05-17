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
    val_loader = None
    val_split = data_cfg.get("val", "val")
    if train_cfg.get("val", True) and val_split and data.get(val_split) is not None:
        val_dataset = YoloDetectionDataset(
            data_cfg["path"],
            split=val_split,
            transforms=ResizeTransform(train_cfg.get("imgsz", 640)),
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=int(train_cfg.get("batch", 16)),
            shuffle=False,
            num_workers=int(train_cfg.get("workers", 8)),
            collate_fn=detection_collate_fn,
            pin_memory=device.type == "cuda",
        )

    optimizer = build_optimizer(adapter.model, cfg["optimizer"], len(train_loader), train_cfg.get("epochs", 100))
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg.get("amp", True)) and device.type == "cuda")

    start_epoch = 0
    best_loss = float("inf")
    best_val_loss = float("inf")
    epochs_without_val_improvement = 0
    patience = int(train_cfg.get("patience", 100))
    resume_path = _resolve_resume_path(train_cfg.get("resume", False), save_dir)
    if resume_path is not None:
        checkpoint = load_checkpoint(resume_path, device)
        _load_resume_state(
            checkpoint=checkpoint,
            adapter=adapter,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_adapter=model_name,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_loss = float(checkpoint.get("best_loss", best_loss))
        best_val_loss = float(checkpoint.get("best_val_loss", best_val_loss))
        epochs_without_val_improvement = int(
            checkpoint.get("epochs_without_val_improvement", epochs_without_val_improvement)
        )
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
            loss_aliases=cfg.get("loss", {}).get("aliases", {}),
            amp=bool(train_cfg.get("amp", True)),
        )
        epoch_stats.update(_build_loss_aliases(epoch_stats, cfg.get("loss", {}).get("aliases", {})))
        if val_loader is not None:
            val_stats = validate_one_epoch(
                adapter=adapter,
                loader=val_loader,
                device=device,
                loss_weights=cfg.get("loss", {}).get("weights", {}),
                loss_aliases=cfg.get("loss", {}).get("aliases", {}),
                amp=bool(train_cfg.get("amp", True)),
            )
            epoch_stats.update({f"val_{key}": value for key, value in val_stats.items()})
        epoch_stats["epoch"] = epoch + 1
        history.append(epoch_stats)

        if train_cfg.get("verbose", True):
            _print_epoch(epoch_stats, optimizer)

        val_loss = epoch_stats.get("val_loss")
        should_stop = False
        if isinstance(val_loss, float) and patience >= 0:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_val_improvement = 0
            else:
                epochs_without_val_improvement += 1
                should_stop = epochs_without_val_improvement >= patience

        is_best_loss = epoch_stats["loss"] < best_loss
        if is_best_loss:
            best_loss = epoch_stats["loss"]

        if train_cfg.get("save", True):
            last_path = save_dir / "weights" / "last.pt"
            save_checkpoint(
                last_path,
                adapter,
                optimizer,
                scheduler,
                epoch,
                cfg,
                names,
                best_loss=best_loss,
                best_val_loss=best_val_loss,
                epochs_without_val_improvement=epochs_without_val_improvement,
            )

            if is_best_loss:
                save_checkpoint(
                    save_dir / "weights" / "best.pt",
                    adapter,
                    optimizer,
                    scheduler,
                    epoch,
                    cfg,
                    names,
                    best_loss=best_loss,
                    best_val_loss=best_val_loss,
                    epochs_without_val_improvement=epochs_without_val_improvement,
                )

            save_period = int(train_cfg.get("save_period", -1))
            if save_period > 0 and (epoch + 1) % save_period == 0:
                save_checkpoint(
                    save_dir / "weights" / f"epoch{epoch + 1}.pt",
                    adapter,
                    optimizer,
                    scheduler,
                    epoch,
                    cfg,
                    names,
                    best_loss=best_loss,
                    best_val_loss=best_val_loss,
                    epochs_without_val_improvement=epochs_without_val_improvement,
                )

        if should_stop:
            break

    return TrainResult({"save_dir": save_dir, "history": history, "config": cfg})


def train_one_epoch(adapter, loader, optimizer, scheduler, scaler, device, loss_weights, loss_aliases, amp=True):
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
            loss = _apply_loss_weights(adapter_loss, losses, loss_weights, loss_aliases)

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


@torch.no_grad()
def validate_one_epoch(adapter, loader, device, loss_weights, loss_aliases, amp=True):
    running = {}
    num_batches = 0

    for images, targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [_move_target_to_device(target, device) for target in targets]

        with torch.amp.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            adapter_loss, losses = adapter.training_step(images, targets)
            loss = _apply_loss_weights(adapter_loss, losses, loss_weights, loss_aliases)

        loss_value = float(loss.detach().cpu())
        num_batches += 1
        running["loss"] = running.get("loss", 0.0) + loss_value
        for key, value in losses.items():
            if torch.is_tensor(value):
                running[key] = running.get(key, 0.0) + float(value.detach().cpu())

    stats = {key: value / max(num_batches, 1) for key, value in running.items()}
    stats.update(_build_loss_aliases(stats, loss_aliases))
    return stats


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


def save_checkpoint(
    path,
    adapter,
    optimizer,
    scheduler,
    epoch,
    cfg,
    names,
    best_loss=None,
    best_val_loss=None,
    epochs_without_val_improvement=0,
):
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
            "best_loss": best_loss,
            "best_val_loss": best_val_loss,
            "epochs_without_val_improvement": epochs_without_val_improvement,
        },
        path,
    )


def load_checkpoint(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return torch.load(path, map_location=device)


def _resolve_resume_path(resume, save_dir):
    if resume is None or resume is False or resume == "":
        return None
    if resume is True:
        return save_dir / "weights" / "last.pt"
    return Path(resume)


def _load_resume_state(checkpoint, adapter, optimizer, scheduler, expected_adapter):
    checkpoint_adapter = checkpoint.get("adapter")
    if checkpoint_adapter is not None and checkpoint_adapter != expected_adapter:
        raise ValueError(
            f"Checkpoint adapter is '{checkpoint_adapter}', but config requested '{expected_adapter}'"
        )

    adapter.model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])


def _apply_loss_weights(total_loss, losses, loss_weights, loss_aliases=None):
    if not loss_weights:
        return total_loss

    weighted = None
    for key, weight in loss_weights.items():
        term = _resolve_loss_term(key, losses, loss_aliases)
        if term is not None:
            term = term * float(weight)
            weighted = term if weighted is None else weighted + term
    return total_loss if weighted is None else weighted


def _resolve_loss_term(key, losses, loss_aliases=None):
    if key in losses and torch.is_tensor(losses[key]):
        return losses[key]

    source_keys = (loss_aliases or {}).get(key, [])
    if isinstance(source_keys, str):
        source_keys = [source_keys]

    term = None
    for source_key in source_keys:
        source_value = losses.get(source_key)
        if torch.is_tensor(source_value):
            term = source_value if term is None else term + source_value

    return term


def _build_loss_aliases(stats, aliases):
    normalized = {}
    for alias, source_keys in (aliases or {}).items():
        if isinstance(source_keys, str):
            source_keys = [source_keys]

        value = 0.0
        matched = False
        for source_key in source_keys or []:
            source_value = stats.get(source_key)
            if isinstance(source_value, (int, float)):
                value += float(source_value)
                matched = True

        if matched:
            normalized[alias] = value

    return normalized


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
    display_keys = ("loss", "cls_loss", "box_loss", "val_loss", "val_cls_loss", "val_box_loss")
    details = " ".join(
        f"{key}={stats[key]:.4f}"
        for key in display_keys
        if isinstance(stats.get(key), float)
    )
    print(f"epoch={stats['epoch']} lr={lr:.6g} {details}")
