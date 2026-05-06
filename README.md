# Friendy Mercury

Friendy Mercury is a planned lightweight object-detection toolkit for this project. The goal is to provide a common training, validation, metrics, and prediction-format layer across multiple detection architectures.

The main idea:

- Keep model-specific code isolated in adapters.
- Keep datasets, metrics, postprocessing, and output formats shared.
- Let different architectures produce comparable YOLO-like predictions.
- Make every model train and validate through the same high-level interface.

This module is currently scaffold-only. It is placed inside the CPPED repository for now and can be moved up one directory later as a standalone package.

## Target Prediction Format

Friendy Mercury should normalize model outputs into one internal prediction format:

```text
x_center y_center width height confidence class_id
```

Labels stay in YOLO format:

```text
class_id x_center y_center width height
```

For raw YOLO outputs, `confidence` is usually:

```text
objectness * class_score
```

For models like RetinaNet, the model score can be used directly as confidence.

## Proposed Structure

```text
friendy_mercury/
├── README.md
├── __init__.py
├── adapters/
│   ├── __init__.py
│   ├── retinanet.py
│   ├── rtdetr.py
│   ├── yolo.py
│   └── yolox.py
├── config.py
├── data.py
├── val.py
├── formats.py
├── metrics.py
├── postprocess.py
├── registry.py
└── train.py
```

## File Responsibilities

### `config.py`

Loads and validates YAML configuration files. This should become the single place that defines required and optional config fields.

### `data.py`

Loads YOLO-format datasets.

Expected label format:

```text
class_id x_center y_center width height
```

This module should eventually provide image discovery, label parsing, dataset splits, a PyTorch `Dataset`, and a detection batch collate function.

### `formats.py`

Defines shared data structures for labels and predictions:

- detection targets
- detection predictions
- normalized `xywh`
- absolute `xyxy`
- conversion helpers

### `postprocess.py`

Contains prediction conversion and filtering logic:

- box format conversion
- confidence thresholding
- non-maximum suppression
- model-output normalization

### `metrics.py`

Computes model-agnostic detection metrics:

- precision
- recall
- mAP50
- mAP50-95
- per-class AP

The same metrics should be used for every architecture so paper comparisons are fair.

### `train.py`

Provides a universal training entry point.

Planned API:

```python
train_from_config("path/to/config.yaml")
```

### `val.py`

Provides a universal validation entry point.

Planned API:

```python
val_from_config("path/to/config.yaml")
```

### `registry.py`

Maps model names to adapter builders.

Example:

```python
MODEL_REGISTRY = {
    "retinanet": build_retinanet,
    "rtdetr": build_rtdetr,
    "yolox": build_yolox,
}
```

### `adapters/`

Each adapter isolates architecture-specific code.

Adapters should handle:

- model construction
- pretrained-weight loading
- native-output conversion
- loss handling if the architecture needs custom training logic

Everything downstream should operate on the same prediction and target formats.

## Development Plan

1. Move shared metric code out of `src/models/test.py` into `friendy_mercury/metrics.py`.
2. Add YOLO-format dataset loading in `friendy_mercury/data.py`.
3. Add box conversion and NMS utilities in `friendy_mercury/postprocess.py`.
4. Implement RetinaNet first because TorchVision has a clean PyTorch detection API.
5. Add RT-DETR and YOLOX adapters.
6. Replace model-specific train/val duplication with Friendy Mercury calls.
