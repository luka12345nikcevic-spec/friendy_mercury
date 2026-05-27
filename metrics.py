from typing import Any, Dict, Iterable, Optional, Sequence

import torch


DEFAULT_IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]


def evaluate_detection(
    predictions: Sequence[torch.Tensor],
    targets: Sequence[Dict[str, Any]],
    iou_thresholds: Optional[Iterable[float]] = None,
    score_threshold: float = 0.001,
    num_classes: Optional[int] = None,
    prediction_classes: Optional[Dict[int, str]] = None,
    target_classes: Optional[Dict[int, str]] = None,
    eval_classes: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Evaluate Friendy-format detection predictions against target dicts.

    When eval_classes is provided, predictions and targets are remapped by class
    name into that evaluation class space. Classes absent from eval_classes are
    ignored, which lets a model trained on extra classes be compared on only the
    classes present in the validation or test dataset.
    """
    thresholds = [float(value) for value in (iou_thresholds or DEFAULT_IOU_THRESHOLDS)]
    if not thresholds:
        raise ValueError('iou_thresholds must contain at least one threshold')

    prepared_targets = [_prepare_target(target) for target in targets]
    prepared_predictions = [
        _prepare_prediction(prediction, target, score_threshold)
        for prediction, target in zip(predictions, prepared_targets)
    ]
    class_ids = _resolve_class_ids(
        prepared_predictions,
        prepared_targets,
        num_classes,
        eval_classes=eval_classes,
    )
    if eval_classes is not None:
        prepared_predictions, prepared_targets = _remap_to_eval_classes(
            prepared_predictions,
            prepared_targets,
            prediction_classes=prediction_classes,
            target_classes=target_classes,
            eval_classes=eval_classes,
        )

    ap_by_threshold = {}
    precision_by_class = {}
    recall_by_class = {}
    f1_by_class = {}
    gt_count_by_class = {}
    pred_count_by_class = {}

    for threshold in thresholds:
        ap_by_threshold[threshold] = {}
        for class_id in class_ids:
            stats = _evaluate_class_at_iou(
                prepared_predictions,
                prepared_targets,
                class_id=class_id,
                iou_threshold=threshold,
            )
            ap_by_threshold[threshold][class_id] = stats['ap']
            if threshold == 0.5:
                precision_by_class[class_id] = stats['precision']
                recall_by_class[class_id] = stats['recall']
                f1_by_class[class_id] = _f1(stats['precision'], stats['recall'])
                gt_count_by_class[class_id] = stats['gt_count']
                pred_count_by_class[class_id] = stats['pred_count']

    ap50_by_class = ap_by_threshold.get(0.5, {})
    ap5095_by_class = {
        class_id: _mean([ap_by_threshold[threshold][class_id] for threshold in thresholds])
        for class_id in class_ids
    }

    precision = _micro_precision(prepared_predictions, prepared_targets, iou_threshold=0.5)
    recall = _micro_recall(prepared_predictions, prepared_targets, iou_threshold=0.5)
    f1 = _f1(precision, recall)

    return {
        'map50': _mean(ap50_by_class.values()),
        'map50_95': _mean(ap5095_by_class.values()),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'num_images': len(targets),
        'num_predictions': int(sum(len(prediction['labels']) for prediction in prepared_predictions)),
        'num_targets': int(sum(len(target['labels']) for target in prepared_targets)),
        'iou_thresholds': thresholds,
        'per_class': {
            int(class_id): {
                'class_name': _class_name(eval_classes, class_id),
                'ap50': ap50_by_class.get(class_id, 0.0),
                'ap50_95': ap5095_by_class.get(class_id, 0.0),
                'precision': precision_by_class.get(class_id, 0.0),
                'recall': recall_by_class.get(class_id, 0.0),
                'f1': f1_by_class.get(class_id, 0.0),
                'ground_truth_count': gt_count_by_class.get(class_id, 0),
                'prediction_count': pred_count_by_class.get(class_id, 0),
            }
            for class_id in class_ids
        },
    }


def _prepare_target(target: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    boxes = target.get('boxes', torch.empty((0, 4)))
    labels = target.get('labels', torch.empty((0,), dtype=torch.long))
    orig_size = target.get('orig_size')
    if orig_size is None:
        if boxes.numel() == 0:
            height, width = 1, 1
        else:
            width = int(torch.max(boxes[:, 2]).item())
            height = int(torch.max(boxes[:, 3]).item())
    elif torch.is_tensor(orig_size):
        height, width = [int(value) for value in orig_size.detach().cpu().flatten()[:2]]
    else:
        height, width = [int(value) for value in orig_size[:2]]

    return {
        'boxes': boxes.detach().cpu().float().reshape(-1, 4),
        'labels': labels.detach().cpu().long().reshape(-1),
        'orig_size': torch.tensor([height, width], dtype=torch.long),
    }


def _prepare_prediction(
    prediction: torch.Tensor,
    target: Dict[str, torch.Tensor],
    score_threshold: float,
) -> Dict[str, torch.Tensor]:
    if prediction is None or prediction.numel() == 0:
        return _empty_prediction()

    prediction = prediction.detach().cpu().float().reshape(-1, 6)
    prediction = prediction[prediction[:, 4] >= float(score_threshold)]
    if prediction.numel() == 0:
        return _empty_prediction()

    height, width = [int(value) for value in target['orig_size']]
    boxes = _xywhn_to_xyxy_tensor(prediction[:, :4], image_width=width, image_height=height)
    return {
        'boxes': boxes,
        'scores': prediction[:, 4],
        'labels': prediction[:, 5].long(),
    }


def _empty_prediction() -> Dict[str, torch.Tensor]:
    return {
        'boxes': torch.empty((0, 4), dtype=torch.float32),
        'scores': torch.empty((0,), dtype=torch.float32),
        'labels': torch.empty((0,), dtype=torch.long),
    }


def _resolve_class_ids(
    predictions,
    targets,
    num_classes: Optional[int],
    eval_classes: Optional[Dict[int, str]] = None,
) -> list[int]:
    if eval_classes is not None:
        return sorted(int(class_id) for class_id in eval_classes)
    if num_classes is not None:
        return list(range(int(num_classes)))

    class_ids = set()
    for target in targets:
        class_ids.update(int(value) for value in target['labels'].tolist())
    for prediction in predictions:
        class_ids.update(int(value) for value in prediction['labels'].tolist())
    return sorted(class_ids)


def _remap_to_eval_classes(
    predictions,
    targets,
    prediction_classes: Optional[Dict[int, str]],
    target_classes: Optional[Dict[int, str]],
    eval_classes: Dict[int, str],
):
    eval_name_to_id = {str(name): int(class_id) for class_id, name in eval_classes.items()}
    prediction_id_to_name = _normalize_class_map(prediction_classes)
    target_id_to_name = _normalize_class_map(target_classes)

    remapped_predictions = [
        _remap_prediction(prediction, prediction_id_to_name, eval_name_to_id)
        for prediction in predictions
    ]
    remapped_targets = [
        _remap_target(target, target_id_to_name, eval_name_to_id)
        for target in targets
    ]
    return remapped_predictions, remapped_targets


def _normalize_class_map(class_map: Optional[Dict[int, str]]) -> Optional[Dict[int, str]]:
    if class_map is None:
        return None
    return {int(class_id): str(name) for class_id, name in class_map.items()}


def _remap_prediction(prediction, id_to_name, eval_name_to_id):
    if id_to_name is None:
        return prediction

    kept_boxes = []
    kept_scores = []
    kept_labels = []
    for box, score, label in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
        class_name = id_to_name.get(int(label))
        if class_name not in eval_name_to_id:
            continue
        kept_boxes.append(box)
        kept_scores.append(score)
        kept_labels.append(eval_name_to_id[class_name])

    return _build_remapped_detection(prediction, kept_boxes, kept_scores, kept_labels)


def _remap_target(target, id_to_name, eval_name_to_id):
    if id_to_name is None:
        return target

    kept_boxes = []
    kept_labels = []
    for box, label in zip(target['boxes'], target['labels']):
        class_name = id_to_name.get(int(label))
        if class_name not in eval_name_to_id:
            continue
        kept_boxes.append(box)
        kept_labels.append(eval_name_to_id[class_name])

    boxes = torch.stack(kept_boxes) if kept_boxes else target['boxes'].new_zeros((0, 4))
    labels = torch.tensor(kept_labels, dtype=torch.long)
    return {
        'boxes': boxes,
        'labels': labels,
        'orig_size': target['orig_size'],
    }


def _build_remapped_detection(reference, boxes, scores, labels):
    if not boxes:
        return _empty_prediction()
    return {
        'boxes': torch.stack(boxes),
        'scores': torch.stack(scores).float(),
        'labels': torch.tensor(labels, dtype=torch.long),
    }


def _class_name(eval_classes: Optional[Dict[int, str]], class_id: int) -> Optional[str]:
    if eval_classes is None:
        return None
    class_name = eval_classes.get(int(class_id))
    if class_name is None:
        return None
    return str(class_name)


def _evaluate_class_at_iou(predictions, targets, class_id: int, iou_threshold: float) -> Dict[str, Any]:
    records = []
    gt_by_image = []
    for image_index, (prediction, target) in enumerate(zip(predictions, targets)):
        target_boxes = target['boxes'][target['labels'] == class_id]
        gt_by_image.append(target_boxes)

        prediction_mask = prediction['labels'] == class_id
        for box, score in zip(prediction['boxes'][prediction_mask], prediction['scores'][prediction_mask]):
            records.append((float(score), image_index, box))

    gt_count = int(sum(len(boxes) for boxes in gt_by_image))
    pred_count = len(records)
    if pred_count == 0:
        return {'ap': 0.0, 'precision': 0.0, 'recall': 0.0, 'gt_count': gt_count, 'pred_count': pred_count}

    records.sort(key=lambda item: item[0], reverse=True)
    matched = [torch.zeros((len(boxes),), dtype=torch.bool) for boxes in gt_by_image]
    true_positives = torch.zeros((pred_count,), dtype=torch.float32)
    false_positives = torch.zeros((pred_count,), dtype=torch.float32)

    for index, (_, image_index, pred_box) in enumerate(records):
        target_boxes = gt_by_image[image_index]
        if len(target_boxes) == 0:
            false_positives[index] = 1.0
            continue

        ious = box_iou(pred_box.reshape(1, 4), target_boxes).reshape(-1)
        best_iou, best_index = torch.max(ious, dim=0)
        if best_iou >= iou_threshold and not matched[image_index][best_index]:
            true_positives[index] = 1.0
            matched[image_index][best_index] = True
        else:
            false_positives[index] = 1.0

    tp_cumsum = torch.cumsum(true_positives, dim=0)
    fp_cumsum = torch.cumsum(false_positives, dim=0)
    precision_curve = tp_cumsum / torch.clamp(tp_cumsum + fp_cumsum, min=1e-12)
    recall_curve = tp_cumsum / max(gt_count, 1)

    return {
        'ap': _average_precision(recall_curve, precision_curve) if gt_count > 0 else 0.0,
        'precision': float(precision_curve[-1].item()),
        'recall': float(recall_curve[-1].item()) if gt_count > 0 else 0.0,
        'gt_count': gt_count,
        'pred_count': pred_count,
    }


def _micro_precision(predictions, targets, iou_threshold: float) -> float:
    tp, fp, _ = _micro_counts(predictions, targets, iou_threshold)
    return float(tp / max(tp + fp, 1))


def _micro_recall(predictions, targets, iou_threshold: float) -> float:
    tp, _, gt = _micro_counts(predictions, targets, iou_threshold)
    return float(tp / max(gt, 1))


def _micro_counts(predictions, targets, iou_threshold: float) -> tuple[int, int, int]:
    tp = 0
    fp = 0
    gt_total = int(sum(len(target['labels']) for target in targets))

    for prediction, target in zip(predictions, targets):
        matched = torch.zeros((len(target['labels']),), dtype=torch.bool)
        order = torch.argsort(prediction['scores'], descending=True)
        for pred_index in order.tolist():
            pred_label = prediction['labels'][pred_index]
            candidate_indices = torch.nonzero(target['labels'] == pred_label, as_tuple=False).flatten()
            if len(candidate_indices) == 0:
                fp += 1
                continue

            ious = box_iou(
                prediction['boxes'][pred_index].reshape(1, 4),
                target['boxes'][candidate_indices],
            ).reshape(-1)
            best_iou, best_local_index = torch.max(ious, dim=0)
            target_index = candidate_indices[best_local_index]
            if best_iou >= iou_threshold and not matched[target_index]:
                tp += 1
                matched[target_index] = True
            else:
                fp += 1

    return tp, fp, gt_total


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return boxes1.new_zeros((boxes1.shape[0], boxes2.shape[0]))

    area1 = torch.clamp(boxes1[:, 2] - boxes1[:, 0], min=0) * torch.clamp(boxes1[:, 3] - boxes1[:, 1], min=0)
    area2 = torch.clamp(boxes2[:, 2] - boxes2[:, 0], min=0) * torch.clamp(boxes2[:, 3] - boxes2[:, 1], min=0)
    top_left = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = torch.clamp(bottom_right - top_left, min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / torch.clamp(union, min=1e-12)


def _xywhn_to_xyxy_tensor(boxes: torch.Tensor, image_width: int, image_height: int) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    x_center, y_center, width, height = boxes.unbind(dim=1)
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return torch.stack([x1, y1, x2, y2], dim=1)


def _average_precision(recall: torch.Tensor, precision: torch.Tensor) -> float:
    if recall.numel() == 0:
        return 0.0

    mrec = torch.cat([recall.new_tensor([0.0]), recall, recall.new_tensor([1.0])])
    mpre = torch.cat([precision.new_tensor([0.0]), precision, precision.new_tensor([0.0])])
    for index in range(mpre.numel() - 1, 0, -1):
        mpre[index - 1] = torch.maximum(mpre[index - 1], mpre[index])
    changing_points = torch.nonzero(mrec[1:] != mrec[:-1], as_tuple=False).flatten()
    ap = torch.sum((mrec[changing_points + 1] - mrec[changing_points]) * mpre[changing_points + 1])
    return float(ap.item())


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / len(values))


def _f1(precision: float, recall: float) -> float:
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return float(2 * precision * recall / denominator)
