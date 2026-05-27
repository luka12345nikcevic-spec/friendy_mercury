from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F

try:
    from ..formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywh
except ImportError:
    from formats import clip_xyxy, xyxy_prediction_to_friendy, xyxy_to_xywh


DEFAULT_YOLOX_VARIANT = "yolox-s"


@dataclass
class YOLOXAdapter:
    model: torch.nn.Module
    num_classes: int
    score_threshold: float = 0.3
    nms_threshold: float = 0.45
    name: str = "yolox"

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
        yolox_targets = self._prepare_targets(targets, images)
        outputs = self.model(batch, yolox_targets)
        losses = {
            key: value
            for key, value in outputs.items()
            if key != "total_loss"
        }
        return outputs["total_loss"], losses

    @torch.no_grad()
    def predict(
        self,
        images,
        score_threshold: Optional[float] = None,
        nms_threshold: Optional[float] = None,
    ):
        try:
            from ..vendor.yolox.utils import postprocess
        except ImportError:
            from vendor.yolox.utils import postprocess

        self.model.eval()
        batch = self._prepare_batch(images)
        outputs = self.model(batch)
        detections = postprocess(
            outputs,
            num_classes=self.num_classes,
            conf_thre=self.score_threshold if score_threshold is None else score_threshold,
            nms_thre=self.nms_threshold if nms_threshold is None else nms_threshold,
        )
        return [
            yolox_detection_to_friendy(detection, image)
            for detection, image in zip(detections, images)
        ]

    def _prepare_batch(self, images):
        device = next(self.model.parameters()).device
        prepared_images = [image.to(device).float() for image in images]
        max_height = _make_divisible(
            max(image.shape[-2] for image in prepared_images),
            32,
        )
        max_width = _make_divisible(
            max(image.shape[-1] for image in prepared_images),
            32,
        )

        padded_images = []
        for image in prepared_images:
            height, width = image.shape[-2:]
            padded_images.append(
                F.pad(
                    image,
                    (0, max_width - width, 0, max_height - height),
                    value=0.0,
                )
            )

        return torch.stack(padded_images)

    def _prepare_targets(self, targets, images):
        device = next(self.model.parameters()).device
        max_objects = max((len(target["labels"]) for target in targets), default=0)
        yolox_targets = torch.zeros(
            (len(targets), max_objects, 5),
            dtype=torch.float32,
            device=device,
        )

        for batch_index, target in enumerate(targets):
            labels = target["labels"].to(device).float()
            boxes = target["boxes"].to(device).float()
            if boxes.numel() == 0:
                continue

            xywh = xyxy_to_xywh(boxes)
            yolox_targets[batch_index, : len(labels), 0] = labels
            yolox_targets[batch_index, : len(labels), 1:5] = xywh

        return yolox_targets


def build_yolox(
    num_classes: int,
    weights: Optional[str] = None,
    variant: str = DEFAULT_YOLOX_VARIANT,
    score_threshold: float = 0.3,
    nms_threshold: float = 0.45,
    **builder_options: Any,
) -> YOLOXAdapter:
    try:
        from ..vendor.yolox.models import build_yolox_model
    except ImportError:
        from vendor.yolox.models import build_yolox_model

    model = build_yolox_model(
        num_classes=num_classes,
        variant=variant,
        **builder_options,
    )
    if weights:
        _load_checkpoint(model, weights)

    return YOLOXAdapter(
        model=model,
        num_classes=num_classes,
        score_threshold=score_threshold,
        nms_threshold=nms_threshold,
    )


def yolox_detection_to_friendy(
    detection: Optional[torch.Tensor],
    image: torch.Tensor,
) -> torch.Tensor:
    if detection is None or detection.numel() == 0:
        return image.new_zeros((0, 6))

    image_height, image_width = image.shape[-2:]
    boxes = clip_xyxy(
        detection[:, 0:4],
        image_width=image_width,
        image_height=image_height,
    )
    confidence = detection[:, 4] * detection[:, 5]

    return xyxy_prediction_to_friendy(
        boxes,
        confidence,
        detection[:, 6],
        image_width=image_width,
        image_height=image_height,
    )


def _make_divisible(value: int, divisor: int) -> int:
    return int((value + divisor - 1) // divisor * divisor)


def _load_checkpoint(model: torch.nn.Module, checkpoint_path: str) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict)
