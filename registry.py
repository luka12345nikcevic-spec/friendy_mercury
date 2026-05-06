try:
    from .adapters.retinanet import build_retinanet
    from .adapters.rtdetr import build_rtdetr
    from .adapters.yolox import build_yolox
except ImportError:
    from adapters.retinanet import build_retinanet
    from adapters.rtdetr import build_rtdetr
    from adapters.yolox import build_yolox


MODEL_REGISTRY = {
    "retinanet": build_retinanet,
    "rtdetr": build_rtdetr,
    "yolox": build_yolox,
}


def build_model(name, **kwargs):
    """Build a registered detector adapter.

    Args:
        name: Registered model name, for example "retinanet".
        **kwargs: Model-specific builder options such as num_classes, weights,
            score_threshold, weights_backbone, trainable_backbone_layers, and
            variant.

    Weight options:
        weights=None: random initialization.
        weights="Default": use a common public pretrained checkpoint.
        weights=<str>: use a model-specific checkpoint path, URL, or model id.

    Default pretrained weights:
        retinanet: Torchvision COCO RetinaNet weights. If num_classes differs
            from COCO, only compatible tensors are loaded.
        rtdetr: Hugging Face PekingU/rtdetr_r50vd.
        yolox: official Megvii YOLOX release weights for the selected variant.

    Examples:
        build_model("retinanet", num_classes=3, weights="Default")
        build_model("retinanet", num_classes=3, weights_backbone="Default")
        build_model("rtdetr", num_classes=3, weights="Default")
        build_model("rtdetr", num_classes=3, weights="PekingU/rtdetr_r50vd")
        build_model("yolox", num_classes=3, variant="yolox-s", weights="Default")
        build_model("yolox", num_classes=3, weights="path/to/checkpoint.pth")
    """
    try:
        builder = MODEL_REGISTRY[name]
    except KeyError as exc:
        available = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model '{name}'. Available models: {available}") from exc

    return builder(**kwargs)
