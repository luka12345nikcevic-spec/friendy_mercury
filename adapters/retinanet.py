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

    model_weights, checkpoint_weights = _resolve_retinanet_weights(
        weight_enum,
        weights,
        num_classes,
    )
    backbone_weights = (
        None
        if model_weights is not None or checkpoint_weights is not None
        else _resolve_torchvision_weights(ResNet50_Weights, weights_backbone)
    )

    model = builder(
        weights=model_weights,
        weights_backbone=backbone_weights,
        num_classes=num_classes,
        trainable_backbone_layers=trainable_backbone_layers,
        **kwargs,
    )

    if checkpoint_weights is not None:
        _load_retinanet_checkpoint(model, checkpoint_weights)

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


def _resolve_retinanet_weights(enum_cls, value, num_classes):
    if value is None or value is False:
        return None, None

    if value is True or (isinstance(value, str) and value.lower() == "default"):
        default_weights = enum_cls.DEFAULT
        default_num_classes = len(default_weights.meta["categories"])
        if num_classes == default_num_classes:
            return default_weights, None
        return None, default_weights

    try:
        return enum_cls.verify(value), None
    except (KeyError, ValueError):
        return None, value


def _resolve_torchvision_weights(enum_cls, value):
    if value is None or value is False:
        return None

    if value is True or (isinstance(value, str) and value.lower() == "default"):
        return enum_cls.DEFAULT

    return enum_cls.verify(value)


def _load_retinanet_checkpoint(model: torch.nn.Module, checkpoint_weights) -> None:
    if hasattr(checkpoint_weights, "get_state_dict"):
        state_dict = checkpoint_weights.get_state_dict(progress=True)
    elif str(checkpoint_weights).startswith(("http://", "https://")):
        state_dict = torch.hub.load_state_dict_from_url(checkpoint_weights, map_location="cpu")
    else:
        checkpoint = torch.load(checkpoint_weights, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)

    _load_matching_state_dict(model, state_dict)


def _load_matching_state_dict(model: torch.nn.Module, state_dict) -> None:
    model_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and model_state[key].shape == value.shape
    }
    model.load_state_dict(compatible_state, strict=False)
