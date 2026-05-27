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

Loads and validates one YAML experiment file. Paths are resolved relative to the YAML file, then expanded into one run per train-dataset/model pair. Validation and test metrics are computed in the eval dataset class space, so train-only classes are ignored when the eval dataset does not list them.

Example:

```yaml
name: helmet-benchmark
output_dir: runs/helmet-benchmark

datasets:
  train:
    - name: field-v1
      images: datasets/field-v1/images/train
      labels: datasets/field-v1/labels/train
      classes: [helmet, head, vest]
    - name: warehouse-v2
      images: datasets/warehouse-v2/images/train
      labels: datasets/warehouse-v2/labels/train
      classes: [helmet, head, vest]
  val:
    name: field-v1-val
    images: datasets/field-v1/images/val
    labels: datasets/field-v1/labels/val
    classes: [helmet, head, vest]
  test:
    name: holdout
    images: datasets/holdout/images/test
    labels: datasets/holdout/labels/test
    classes: [helmet, head, vest]

# Class IDs can differ between train and val/test as long as class names match.
# Metrics use the val/test classes and ignore predictions for train-only names.
# num_classes: auto makes each run use len(current train dataset classes).

models:
  - name: retinanet
    num_classes: auto
    weights_backbone: DEFAULT
  - name: yolox
    num_classes: auto
    variant: yolox-s
  - name: rtdetr
    num_classes: auto
    weights: PekingU/rtdetr_r50vd

training:
  epochs: 100
  batch_size: 4
  num_workers: 4
  device: auto
  amp: false
  optimizer:
    name: adamw
    lr: 0.0001
    weight_decay: 0.0001
  scheduler: null

evaluation:
  batch_size: 4
  score_threshold: 0.001
  iou_thresholds: [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
```

### `data.py`

Loads YOLO-format image/label directory datasets.

Expected label format:

```text
class_id x_center y_center width height
```

This module provides image discovery, YOLO label parsing, a PyTorch `Dataset`, and a detection batch collate function.

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

Provides the universal config-driven training entry point. Each entry under `datasets.train` is treated as its own training dataset, and every configured model is trained once per dataset. For example, two train datasets and three models produce six independent runs. It saves `last.pt`, saves `best.pt` when validation is configured, writes per-run history/results YAML files, and saves raw test predictions when a test dataset is configured.

API:

```python
from friendy_mercury.train import train_from_config

results = train_from_config("configs/experiment.yaml")
```

CLI:

```bash
python train.py configs/experiment.yaml
```

Detection metrics are still pending in `metrics.py`, so `best.pt` is chosen by validation loss when a validation dataset exists. Without validation, only `last.pt` is saved.

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
