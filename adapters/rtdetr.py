from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F


DEFAULT_RTDETR_WEIGHTS = "PekingU/rtdetr_r50vd"


@dataclass
class RTDETRAdapter:
    model: torch.nn.Module
    image_processor: Any
    num_classes: int
    score_threshold: float = 0.5
    image_mean: tuple = (0.485, 0.456, 0.406)
    image_std: tuple = (0.229, 0.224, 0.225)
    name: str = "rtdetr"

    def to(self, device):
        self.model.to(device)
        return self

    def train(self, mode: bool = True):
        self.model.train(mode)
        return self

    def eval(self):
        self.model.eval()
        return self

    def training_step(self, images, targets):
        self.model.train()
        batch = self._prepare_batch(images)
        labels = self._prepare_labels(targets, images)
        outputs = self.model(**batch, labels=labels)
        losses = (
            dict(outputs.loss_dict)
            if outputs.loss_dict is not None
            else {"loss": outputs.loss}
        )
        return outputs.loss, losses

    @torch.no_grad()
    def predict(self, images, score_threshold: Optional[float] = None):
        self.model.eval()
        batch = self._prepare_batch(images)
        outputs = self.model(**batch)
        target_sizes = torch.tensor(
            [[image.shape[-2], image.shape[-1]] for image in images],
            dtype=torch.long,
            device=batch["pixel_values"].device,
        )
        predictions = self.image_processor.post_process_object_detection(
            outputs,
            threshold=self.score_threshold if score_threshold is None else score_threshold,
            target_sizes=target_sizes,
            use_focal_loss=getattr(self.model.config, "use_focal_loss", True),
        )
        return [
            rtdetr_prediction_to_friendy(prediction, image)
            for prediction, image in zip(predictions, images)
        ]

    def _prepare_batch(self, images):
        device = next(self.model.parameters()).device
        image_mean = torch.tensor(self.image_mean, device=device).view(3, 1, 1)
        image_std = torch.tensor(self.image_std, device=device).view(3, 1, 1)

        prepared_images = [
            ((image.to(device).float() - image_mean) / image_std)
            for image in images
        ]
        max_height = max(image.shape[-2] for image in prepared_images)
        max_width = max(image.shape[-1] for image in prepared_images)

        pixel_values = []
        pixel_masks = []
        for image in prepared_images:
            height, width = image.shape[-2:]
            pixel_values.append(
                F.pad(image, (0, max_width - width, 0, max_height - height))
            )

            mask = torch.zeros((max_height, max_width), dtype=torch.long, device=device)
            mask[:height, :width] = 1
            pixel_masks.append(mask)

        return {
            "pixel_values": torch.stack(pixel_values),
            "pixel_mask": torch.stack(pixel_masks),
        }

    def _prepare_labels(self, targets, images):
        device = next(self.model.parameters()).device
        labels = []
        for target, image in zip(targets, images):
            image_height, image_width = image.shape[-2:]
            boxes = target["boxes"].to(device).float()
            labels.append(
                {
                    "class_labels": target["labels"].to(device).long(),
                    "boxes": _xyxy_to_xywhn(
                        boxes,
                        image_width=image_width,
                        image_height=image_height,
                    ),
                }
            )
        return labels


def build_rtdetr(
    num_classes: int,
    weights: Optional[str] = None,
    score_threshold: float = 0.5,
    image_mean: tuple = (0.485, 0.456, 0.406),
    image_std: tuple = (0.229, 0.224, 0.225),
    ignore_mismatched_sizes: bool = True,
    **config_kwargs: Any,
) -> RTDETRAdapter:
    (
        RTDetrConfig,
        RTDetrForObjectDetection,
        RTDetrImageProcessor,
    ) = _load_transformers_rtdetr()

    id2label = config_kwargs.pop(
        "id2label",
        {class_id: str(class_id) for class_id in range(num_classes)},
    )
    label2id = config_kwargs.pop(
        "label2id",
        {class_name: class_id for class_id, class_name in id2label.items()},
    )

    if weights is True:
        weights = DEFAULT_RTDETR_WEIGHTS
    elif weights is False:
        weights = None

    if weights is None:
        config = RTDetrConfig(
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            **config_kwargs,
        )
        model = RTDetrForObjectDetection(config)
    else:
        model = RTDetrForObjectDetection.from_pretrained(
            weights,
            num_labels=num_classes,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=ignore_mismatched_sizes,
            **config_kwargs,
        )

    return RTDETRAdapter(
        model=model,
        image_processor=RTDetrImageProcessor(),
        num_classes=num_classes,
        score_threshold=score_threshold,
        image_mean=image_mean,
        image_std=image_std,
    )


def rtdetr_prediction_to_friendy(
    prediction: Dict[str, torch.Tensor], image: torch.Tensor
) -> torch.Tensor:
    boxes = prediction["boxes"]
    scores = prediction["scores"]
    labels = prediction["labels"]

    if boxes.numel() == 0:
        return boxes.new_zeros((0, 6))

    image_height, image_width = image.shape[-2:]
    xywhn = _xyxy_to_xywhn(boxes, image_width=image_width, image_height=image_height)

    return torch.cat(
        [
            xywhn,
            scores.reshape(-1, 1).to(dtype=boxes.dtype),
            labels.reshape(-1, 1).to(dtype=boxes.dtype),
        ],
        dim=1,
    )


def _xyxy_to_xywhn(
    boxes: torch.Tensor, image_width: int, image_height: int
) -> torch.Tensor:
    if boxes.numel() == 0:
        return boxes.reshape(-1, 4)

    x1, y1, x2, y2 = boxes.unbind(dim=1)
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    normalizer = boxes.new_tensor(
        [image_width, image_height, image_width, image_height]
    )
    return torch.stack([x_center, y_center, width, height], dim=1) / normalizer


def _load_transformers_rtdetr():
    try:
        from transformers import (
            RTDetrConfig,
            RTDetrForObjectDetection,
            RTDetrImageProcessor,
        )
    except ImportError as exc:
        raise ImportError(
            "RT-DETR requires optional dependencies. "
            "Install it with `pip install -r requirements-rtdetr.txt`."
        ) from exc

    return RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor
