from pathlib import Path
import csv
import time

import torch
from torch.utils.data import DataLoader

try:
    from .config import load_config
    from .data import YoloDetectionDataset, detection_collate_fn, load_data_yaml
    from .registry import build_model
    from .train import ResizeTransform, load_checkpoint
except ImportError:
    from config import load_config
    from data import YoloDetectionDataset, detection_collate_fn, load_data_yaml
    from registry import build_model
    from train import ResizeTransform, load_checkpoint


class ValResult(dict):
    @property
    def metrics(self):
        return self["metrics"]


def val_from_config(config_path=None, **overrides):
    cfg = load_config(config_path, overrides=overrides or None)
    return val(cfg)


def val(cfg):
    train_cfg = cfg["train"]
    data_cfg = cfg["data"]
    eval_cfg = cfg.get("eval", {})
    model_cfg = dict(cfg["model"])

    if data_cfg.get("path") is None:
        raise ValueError("Validation config requires data.path pointing to a YOLO data.yaml")

    data = load_data_yaml(data_cfg["path"])
    test_split = data_cfg.get("test") or "test"
    if data.get(test_split) is None:
        raise ValueError(f"Validation requires data.yaml split '{test_split}'")

    device = _select_device(train_cfg.get("device"))
    checkpoint_path = _resolve_eval_weights(
        eval_cfg.get("weights", "best.pt"),
        train_cfg.get("project", "runs/train"),
        train_cfg.get("name", "exp"),
    )
    checkpoint = load_checkpoint(checkpoint_path, device)

    names = checkpoint.get("names", data["names"])
    num_classes = model_cfg.pop("num_classes", None) or len(names)
    model_name = model_cfg.pop("name")
    checkpoint_adapter = checkpoint.get("adapter")
    if checkpoint_adapter is not None and checkpoint_adapter != model_name:
        raise ValueError(
            f"Checkpoint adapter is '{checkpoint_adapter}', but config requested '{model_name}'"
        )

    model_cfg["num_classes"] = num_classes
    model_cfg["weights"] = None
    model_cfg["weights_backbone"] = None
    model_cfg = {key: value for key, value in model_cfg.items() if value is not None}
    adapter = build_model(model_name, **model_cfg).to(device)
    adapter.model.load_state_dict(checkpoint["model"])
    adapter.eval()

    dataset = YoloDetectionDataset(
        data_cfg["path"],
        split=test_split,
        transforms=ResizeTransform(train_cfg.get("imgsz", 640)),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("batch", 16)),
        shuffle=False,
        num_workers=int(train_cfg.get("workers", 8)),
        collate_fn=detection_collate_fn,
        pin_memory=device.type == "cuda",
    )

    predictions, targets, speed = _collect_predictions(
        adapter=adapter,
        loader=loader,
        device=device,
        conf=float(eval_cfg.get("conf", 0.001)),
        max_det=int(eval_cfg.get("max_det", 300)),
    )
    metrics = compute_detection_metrics(
        predictions=predictions,
        targets=targets,
        num_classes=num_classes,
        iou_threshold=float(eval_cfg.get("iou", 0.7)),
    )
    metrics["speed"] = speed
    metrics_path = _save_metrics_csv(
        checkpoint_path=checkpoint_path,
        metrics=metrics,
        model_name=model_name,
        eval_cfg=eval_cfg,
    )

    return ValResult(
        {
            "checkpoint": checkpoint_path,
            "metrics": metrics,
            "metrics_path": metrics_path,
            "config": cfg,
            "names": names,
        }
    )


@torch.no_grad()
def _collect_predictions(adapter, loader, device, conf, max_det):
    predictions = []
    targets = []
    inference_time = 0.0
    num_images = 0

    for images, batch_targets in loader:
        images = [image.to(device, non_blocking=True) for image in images]
        _sync_device(device)
        start = time.perf_counter()
        batch_predictions = adapter.predict(images)
        _sync_device(device)
        inference_time += time.perf_counter() - start
        num_images += len(images)

        for prediction, target, image in zip(batch_predictions, batch_targets, images):
            predictions.append(_filter_prediction(prediction.detach().cpu(), conf, max_det))
            targets.append(_target_to_friendy(target, image.shape[-1], image.shape[-2]))

    return predictions, targets, _speed_metrics(inference_time, num_images)


def _speed_metrics(inference_time, num_images):
    if num_images <= 0:
        return {
            "inference_time_s": 0.0,
            "inference_ms_per_image": 0.0,
            "images_per_second": 0.0,
        }

    return {
        "inference_time_s": inference_time,
        "inference_ms_per_image": inference_time * 1000.0 / num_images,
        "images_per_second": num_images / inference_time if inference_time > 0 else 0.0,
    }


def _sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def compute_detection_metrics(predictions, targets, num_classes, iou_threshold=0.7):
    thresholds = [0.5 + index * 0.05 for index in range(10)]
    per_class = {}
    ap50_values = []
    ap5095_values = []
    precision_values = []
    recall_values = []

    for class_id in range(num_classes):
        class_result = _evaluate_class(predictions, targets, class_id, iou_threshold)
        per_class[class_id] = class_result
        precision_values.append(class_result["precision"])
        recall_values.append(class_result["recall"])

        ap50 = _average_precision_for_class(predictions, targets, class_id, 0.5)
        ap50_values.append(ap50)
        ap_thresholds = [
            _average_precision_for_class(predictions, targets, class_id, threshold)
            for threshold in thresholds
        ]
        ap5095_values.append(_mean(ap_thresholds))
        per_class[class_id]["ap50"] = ap50
        per_class[class_id]["ap50_95"] = ap5095_values[-1]

    return {
        "precision": _mean(precision_values),
        "recall": _mean(recall_values),
        "map50": _mean(ap50_values),
        "map50_95": _mean(ap5095_values),
        "per_class": per_class,
        "num_images": len(targets),
        "num_predictions": sum(len(prediction) for prediction in predictions),
        "num_targets": sum(len(target) for target in targets),
    }


def _evaluate_class(predictions, targets, class_id, iou_threshold):
    true_positives, false_positives, num_targets = _match_class_predictions(
        predictions,
        targets,
        class_id,
        iou_threshold,
    )
    tp = sum(true_positives)
    fp = sum(false_positives)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(num_targets, 1)
    return {
        "precision": precision,
        "recall": recall,
        "num_targets": num_targets,
        "num_predictions": tp + fp,
    }


def _average_precision_for_class(predictions, targets, class_id, iou_threshold):
    true_positives, false_positives, num_targets = _match_class_predictions(
        predictions,
        targets,
        class_id,
        iou_threshold,
    )
    if num_targets == 0:
        return 0.0

    tp_cumsum = 0
    fp_cumsum = 0
    precisions = []
    recalls = []
    for tp, fp in zip(true_positives, false_positives):
        tp_cumsum += tp
        fp_cumsum += fp
        precisions.append(tp_cumsum / max(tp_cumsum + fp_cumsum, 1))
        recalls.append(tp_cumsum / num_targets)

    return _compute_ap(recalls, precisions)


def _match_class_predictions(predictions, targets, class_id, iou_threshold):
    candidates = []
    class_targets = {}
    num_targets = 0

    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        target_mask = target[:, 5] == class_id if len(target) else torch.zeros((0,), dtype=torch.bool)
        target_boxes = target[target_mask, :4]
        class_targets[image_index] = {
            "boxes": target_boxes,
            "matched": torch.zeros((len(target_boxes),), dtype=torch.bool),
        }
        num_targets += len(target_boxes)

        if len(prediction):
            prediction_mask = prediction[:, 5] == class_id
            for pred in prediction[prediction_mask]:
                candidates.append((float(pred[4]), image_index, pred[:4]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    true_positives = []
    false_positives = []

    for _, image_index, pred_box in candidates:
        target_info = class_targets[image_index]
        target_boxes = target_info["boxes"]
        if len(target_boxes) == 0:
            true_positives.append(0)
            false_positives.append(1)
            continue

        ious = box_iou(pred_box.reshape(1, 4), target_boxes).squeeze(0)
        best_iou, best_index = ious.max(dim=0)
        if best_iou >= iou_threshold and not target_info["matched"][best_index]:
            target_info["matched"][best_index] = True
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    return true_positives, false_positives, num_targets


def box_iou(boxes1, boxes2):
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((len(boxes1), len(boxes2)))

    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (right_bottom - left_top).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-12)


def _compute_ap(recalls, precisions):
    if not recalls:
        return 0.0

    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])

    ap = 0.0
    for index in range(1, len(mrec)):
        if mrec[index] != mrec[index - 1]:
            ap += (mrec[index] - mrec[index - 1]) * mpre[index]
    return ap


def _filter_prediction(prediction, conf, max_det):
    if len(prediction) == 0:
        return prediction.reshape(0, 6)

    prediction = prediction[prediction[:, 4] >= conf]
    if len(prediction) == 0:
        return prediction.reshape(0, 6)

    order = prediction[:, 4].argsort(descending=True)
    prediction = prediction[order]
    prediction = prediction[:max_det].clone()
    prediction[:, :4] = _xywh_to_xyxy(prediction[:, :4])
    return prediction


def _target_to_friendy(target, image_width, image_height):
    boxes = target["boxes"].detach().cpu()
    labels = target["labels"].detach().cpu()
    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6))

    normalizer = boxes.new_tensor([image_width, image_height, image_width, image_height])
    boxes = boxes / normalizer
    return torch.cat(
        [
            boxes,
            boxes.new_ones((len(boxes), 1)),
            labels.reshape(-1, 1).to(dtype=boxes.dtype),
        ],
        dim=1,
    )


def _xywh_to_xyxy(boxes):
    x_center, y_center, width, height = boxes.unbind(dim=1)
    half_width = width / 2
    half_height = height / 2
    return torch.stack(
        [
            x_center - half_width,
            y_center - half_height,
            x_center + half_width,
            y_center + half_height,
        ],
        dim=1,
    )


def _resolve_eval_weights(weights, project, name):
    if weights is None or weights is False or weights == "":
        weights = "best.pt"

    path = Path(weights)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return Path(project) / name / "weights" / path


def _save_metrics_csv(checkpoint_path, metrics, model_name, eval_cfg):
    path = checkpoint_path.parent.parent / "val_metrics.csv"
    row = _metrics_csv_row(checkpoint_path, metrics, model_name, eval_cfg)
    write_header = not path.exists()
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return path


def _metrics_csv_row(checkpoint_path, metrics, model_name, eval_cfg):
    speed = metrics.get("speed", {})
    return {
        "checkpoint": str(checkpoint_path),
        "model": model_name,
        "eval_conf": float(eval_cfg.get("conf", 0.001)),
        "eval_iou": float(eval_cfg.get("iou", 0.7)),
        "eval_max_det": int(eval_cfg.get("max_det", 300)),
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "map50": metrics["map50"],
        "map50_95": metrics["map50_95"],
        "num_images": metrics["num_images"],
        "num_predictions": metrics["num_predictions"],
        "num_targets": metrics["num_targets"],
        "inference_time_s": speed.get("inference_time_s", 0.0),
        "inference_ms_per_image": speed.get("inference_ms_per_image", 0.0),
        "images_per_second": speed.get("images_per_second", 0.0),
    }


def _select_device(device):
    if device is None or device == "":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _mean(values):
    return sum(values) / max(len(values), 1)
