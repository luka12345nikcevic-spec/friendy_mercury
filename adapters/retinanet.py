from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class RetinaNetAdapter:
    model: torch.nn.Module
    num_classes: int
    name: str = "retinanet"

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
        losses = self.model(images, targets)
        total_loss = sum(loss for loss in losses.values())
        return total_loss, losses

    @torch.no_grad()
    def predict(self, images):
        self.model.eval()
        predictions = self.model(images)
        return [
            retinanet_prediction_to_friendy(prediction, image)
            for prediction, image in zip(predictions, images)
        ]


def build_retinanet(
    num_classes: int,
    weights: Optional[str] = None,
    weights_backbone: Optional[str] = None,
    trainable_backbone_layers: Optional[int] = None,
    variant: str = "resnet50_fpn_v2",
    **kwargs: Any,
) -> RetinaNetAdapter:
    if variant not in {"resnet50_fpn", "resnet50_fpn_v2"}:
        raise ValueError(f"Unsupported RetinaNet variant: {variant}")

    from torchvision.models.detection import (
        RetinaNet_ResNet50_FPN_V2_Weights,
        RetinaNet_ResNet50_FPN_Weights,
        retinanet_resnet50_fpn,
        retinanet_resnet50_fpn_v2,
    )
    from torchvision.models import ResNet50_Weights

    if variant == "resnet50_fpn_v2":
        builder = retinanet_resnet50_fpn_v2
        weight_enum = RetinaNet_ResNet50_FPN_V2_Weights
    else:
        builder = retinanet_resnet50_fpn
        weight_enum = RetinaNet_ResNet50_FPN_Weights

    model_weights = _resolve_weights(weight_enum, weights)
    backbone_weights = (
        None
        if model_weights is not None
        else _resolve_weights(ResNet50_Weights, weights_backbone)
    )

    model = builder(
        weights=model_weights,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )
    return RetinaNetAdapter(model=model, num_classes=num_classes)


def retinanet_prediction_to_friendy(
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
    x1, y1, x2, y2 = boxes.unbind(dim=1)
    width = x2 - x1
    height = y2 - y1
    x_center = x1 + width / 2
    y_center = y1 + height / 2

    normalizer = boxes.new_tensor(
        [image_width, image_height, image_width, image_height]
    )
    return torch.stack([x_center, y_center, width, height], dim=1) / normalizer


def _resolve_weights(enum_cls, value):
    if value is None:
        return None

    if value is True:
        value = "DEFAULT"
    elif value is False:
        return None

    return enum_cls.verify(value)
