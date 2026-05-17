# Training Configuration

`train.py` trains a detector from a YAML config through:

```python
from train import train_from_config

result = train_from_config("config/retinanet.yaml", data={"path": "path/to/data.yaml"})
print(result.save_dir)
```

`load_config()` always starts from `config/default.yaml`, deep-merges
`config/loss_aliases.yaml`, then deep-merges the config file you pass, then
deep-merges any Python keyword overrides. This means model presets such as
`config/retinanet.yaml`, `config/rtdetr.yaml`, and `config/yolox.yaml` only need
to define values that differ from the defaults.

## Minimal Config

`data.path` is the only required field that is not usable from the default
config as-is.

```yaml
data:
  path: /datasets/my_dataset/data.yaml
```

The dataset file must be YOLO-style and include `names` plus the split paths
used by `data.train` and, when training validation is enabled, `data.val`.

Example `data.yaml`:

```yaml
path: /datasets/my_dataset
train: images/train
val: images/val
names:
  0: class_a
  1: class_b
```

## Full Config Shape

This is the default config shape plus the automatically loaded loss aliases from
`config/loss_aliases.yaml`. Some fields are defined for compatibility or future
behavior but are not currently read by `train.py`; those are listed in
[Defined But Not Used By Training](#defined-but-not-used-by-training).

```yaml
model:
  name: retinanet
  variant: resnet50_fpn_v2
  weights: null
  weights_backbone: null
  trainable_backbone_layers: null
  num_classes: null

data:
  path: null
  train: train
  val: val

train:
  epochs: 100
  patience: 100
  batch: 16
  imgsz: 640
  device: null
  workers: 8
  project: runs/train
  name: exp
  exist_ok: false
  seed: 0
  deterministic: true
  resume: false
  amp: true
  fraction: 1.0
  freeze: null
  save: true
  save_period: -1
  val: true
  verbose: true

optimizer:
  name: auto
  lr0: 0.01
  lrf: 0.01
  momentum: 0.937
  weight_decay: 0.0005
  nesterov: true

scheduler:
  cos_lr: false
  warmup_epochs: 3.0
  warmup_momentum: 0.8

loss:
  weights: {}
  aliases:
    cls_loss:
      - classification
      - cls_loss
      - class_loss
      - loss_cls
      - loss_class
      - loss_ce
      - loss_vfl
      - loss_labels
    box_loss:
      - bbox_regression
      - box_loss
      - iou_loss
      - giou_loss
      - l1_loss
      - loss_box
      - loss_iou
      - loss_giou
      - loss_l1
      - loss_bbox
    obj_loss:
      - obj_loss
      - objectness_loss
      - conf_loss
      - confidence_loss
      - loss_obj
      - loss_objectness
      - loss_conf
    dfl_loss:
      - dfl_loss
      - loss_dfl

augment:
  hsv_h: 0.015
  hsv_s: 0.7
  hsv_v: 0.4
  degrees: 0.0
  translate: 0.1
  scale: 0.5
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.5
  mosaic: 0.0
  mixup: 0.0
  copy_paste: 0.0
  close_mosaic: 10
```

## Model

`model.name` selects the adapter from `registry.py`.

Supported values:

- `retinanet`
- `rtdetr`
- `yolox`

Common model keys:

| Key | Default | Used by | Description |
| --- | --- | --- | --- |
| `name` | `retinanet` | all | Registered adapter name. |
| `num_classes` | `null` | all | If `null`, inferred from `data.yaml` `names`. |
| `weights` | `null` | all | `null`/`false` for random init, `Default`/`true` for adapter default pretrained weights, or a model-specific path, URL, or model id. |
| any extra key | varies | adapter-specific | After `name` and `num_classes` are handled, non-null model keys are passed into the selected adapter builder. |

### RetinaNet

Preset: `config/retinanet.yaml`

```yaml
model:
  name: retinanet
  variant: resnet50_fpn_v2
  weights: Default
  weights_backbone: null
  trainable_backbone_layers: 3
```

Supported RetinaNet keys:

| Key | Values | Description |
| --- | --- | --- |
| `variant` | `resnet50_fpn`, `resnet50_fpn_v2` | TorchVision RetinaNet variant. |
| `weights` | `null`, `false`, `true`, `Default`, TorchVision weight enum value, local path, URL | Full-model weights. When default COCO weights have a different class count, only compatible tensors are loaded. |
| `weights_backbone` | `null`, `false`, `true`, `Default`, TorchVision `ResNet50_Weights` enum value | Backbone weights. Ignored when full-model weights or checkpoint weights are used. |
| `trainable_backbone_layers` | `null` or integer | Passed to TorchVision RetinaNet builder. |
| extra keys | TorchVision builder kwargs | Forwarded to `retinanet_resnet50_fpn*`. |

RetinaNet uses the universal loss weight keys:

```yaml
loss:
  weights:
    cls_loss: 1.0
    box_loss: 1.0
```

With the default `loss.aliases`, RetinaNet epoch logs also include:

```text
cls_loss = classification
box_loss = bbox_regression
```

### RT-DETR

Preset: `config/rtdetr.yaml`

```yaml
model:
  name: rtdetr
  weights: Default
  score_threshold: 0.5
```

Supported RT-DETR keys:

| Key | Values | Description |
| --- | --- | --- |
| `weights` | `null`, `false`, `true`, `Default`, Hugging Face model id/path | `Default` and `true` resolve to `PekingU/rtdetr_r50vd`. |
| `score_threshold` | float | Stored on the adapter for prediction. It does not affect training loss. |
| `image_mean` | 3-number sequence | Normalization mean used by the adapter. |
| `image_std` | 3-number sequence | Normalization std used by the adapter. |
| `ignore_mismatched_sizes` | bool | Passed to `RTDetrForObjectDetection.from_pretrained`. Defaults to `true` in the adapter. |
| `id2label` | mapping | Optional label mapping passed to Hugging Face config/model. |
| `label2id` | mapping | Optional label mapping passed to Hugging Face config/model. |
| extra keys | `RTDetrConfig` or `from_pretrained` kwargs | Forwarded to the Hugging Face model/config constructor. |

RT-DETR uses the universal loss weight keys:

```yaml
loss:
  weights:
    cls_loss: 1.0
    box_loss: 1.0
```

For RT-DETR, the displayed loss terms come from Hugging Face
`outputs.loss_dict`. Depending on the installed `transformers` implementation
and config, classification is commonly exposed as `loss_vfl` or `loss_ce`, and
box regression is commonly exposed as `loss_bbox` and `loss_giou`. The default
`loss.aliases` maps those names to:

```text
cls_loss = loss_ce or loss_vfl
box_loss = loss_bbox + loss_giou
```

### YOLOX

Preset: `config/yolox.yaml`

```yaml
model:
  name: yolox
  variant: yolox-s
  weights: Default
  score_threshold: 0.3
  nms_threshold: 0.45
```

Supported YOLOX keys:

| Key | Values | Description |
| --- | --- | --- |
| `variant` | `yolox-nano`, `yolox-tiny`, `yolox-s`, `yolox-m`, `yolox-l`, `yolox-x` | YOLOX model size. |
| `weights` | `null`, `false`, `true`, `Default`, local path, URL | `Default` and `true` resolve to the official Megvii release URL for the selected variant. |
| `score_threshold` | float | Stored on the adapter for prediction. It does not affect training loss. |
| `nms_threshold` | float | Stored on the adapter for prediction. It does not affect training loss. |
| extra keys | YOLOX builder options | Forwarded to `vendor.yolox.models.build_yolox_model`. |

YOLOX uses the universal loss weight keys:

```yaml
loss:
  weights:
    cls_loss: 1.0
    box_loss: 1.0
    obj_loss: 1.0
```

With the default `loss.aliases`, YOLOX epoch logs also include:

```text
cls_loss = cls_loss
box_loss = iou_loss + l1_loss
```

## Data

| Key | Default | Required | Description |
| --- | --- | --- | --- |
| `path` | `null` | yes | Path to YOLO `data.yaml`. Training raises `ValueError` when this is missing. |
| `train` | `train` | yes | Name of the split key inside the dataset YAML. `YoloDetectionDataset` uses `data_yaml[train]` to find images. |
| `val` | `val` | no | Name of the validation split key inside the dataset YAML. Used when `train.val: true` and the dataset YAML defines that split. |

The selected split may be a directory, a `.txt` file of image paths, or a list of
entries. Image labels are resolved by replacing an `images` path component with
`labels` and changing the suffix to `.txt`.

For checkpoint evaluation on the test split, see `VAL.md`.

## Validation During Training

When `train.val: true`, `train.py` looks for the split named by `data.val`
inside the loaded YOLO `data.yaml`. With the default config, that means the
dataset YAML should define:

```yaml
val: images/val
```

If the validation split exists, a validation dataloader is created and a
loss-only validation pass runs after every training epoch. Validation uses the
same adapter loss path as training, but runs with gradients disabled.

Validation stats are added to `epoch_stats` with a `val_` prefix. For example:

```text
val_loss
val_cls_loss
val_box_loss
```

The epoch printer only displays:

```text
epoch=1 lr=... loss=... cls_loss=... box_loss=... val_loss=... val_cls_loss=... val_box_loss=...
```

`train.patience` enables early stopping from validation loss. After each
validation pass, training tracks the lowest `val_loss` seen so far. If `val_loss`
does not improve for `train.patience` consecutive epochs, the loop stops after
the current epoch. A value of `0` stops on the first non-improving validation
epoch. If validation is disabled or no validation split exists, patience has no
effect.

## Train

| Key | Default | Description |
| --- | --- | --- |
| `epochs` | `100` | Number of epochs. Also used to decide `optimizer.name: auto`. |
| `patience` | `100` | Early-stopping patience in epochs, based on `val_loss`. Ignored when validation does not run. |
| `batch` | `16` | Dataloader batch size. |
| `imgsz` | `640` | Square resize size applied to every training image. |
| `device` | `null` | Torch device string. `null` or empty string selects `cuda:0` if available, otherwise `cpu`. |
| `workers` | `8` | Dataloader worker count. |
| `project` | `runs/train` | Parent directory for run outputs. |
| `name` | `exp` | Run directory name below `project`. |
| `exist_ok` | `false` | If `false`, existing run directories are auto-incremented (`exp2`, `exp3`, ...). |
| `seed` | `0` | Python and Torch seed. Also controls dataset fraction shuffling. |
| `deterministic` | `true` | Enables deterministic Torch algorithms with warnings only and disables CuDNN benchmark. |
| `amp` | `true` | Enables CUDA autocast and `GradScaler` when the selected device is CUDA. |
| `fraction` | `1.0` | Fraction of the training dataset to keep. Values below `1.0` select a seeded subset. |
| `freeze` | `null` | `null`/`false` leaves all parameters trainable. An integer freezes the first N parameters. A list freezes those parameter indexes. |
| `save` | `true` | Saves checkpoints when enabled. |
| `save_period` | `-1` | If greater than 0, saves `epoch{N}.pt` every N epochs. |
| `resume` | `false` | `false` starts fresh, `true` resumes from `{project}/{name}/weights/last.pt`, and a string path resumes from that checkpoint. |
| `val` | `true` | Runs a loss-only validation pass each epoch when `data.val` points to a split present in the dataset YAML. |
| `verbose` | `true` | Prints one-line epoch stats. |

Checkpoint files are written below:

```text
{project}/{name}/weights/last.pt
{project}/{name}/weights/best.pt
{project}/{name}/weights/epoch{N}.pt
```

Each checkpoint contains:

- `epoch`
- `model`
- `optimizer`
- `scheduler`
- `config`
- `names`
- `adapter`
- `best_loss`
- `best_val_loss`
- `epochs_without_val_improvement`

## Resume

`train.resume` restores a full training checkpoint and continues from the next
epoch. This is different from `model.weights`, which only initializes model
weights for a fresh fine-tuning run.

```yaml
train:
  resume: false
```

Starts a new run.

```yaml
train:
  project: runs/train
  name: exp
  resume: true
```

Resumes from:

```text
runs/train/exp/weights/last.pt
```

```yaml
train:
  resume: runs/train/exp/weights/last.pt
```

Resumes from an explicit checkpoint path.

Resume restores:

- model weights
- optimizer state
- scheduler state
- next epoch index
- best training loss
- best validation loss
- validation patience counter

## Optimizer

| Key | Default | Description |
| --- | --- | --- |
| `name` | `auto` | One of `auto`, `SGD`, `Adam`, `AdamW`, `NAdam`, `RAdam`, `RMSProp`, or `RMS`. Case-insensitive. |
| `lr0` | `0.01` | Initial learning rate. |
| `lrf` | `0.01` | Final LR multiplier used by the scheduler. |
| `momentum` | `0.937` | Momentum for SGD/RMSProp or first beta for Adam-family optimizers. |
| `weight_decay` | `0.0005` | Optimizer weight decay. |
| `nesterov` | `true` | Used only by SGD. |

`auto` chooses:

- `SGD` when `len(train_loader) * epochs > 10000`
- `AdamW` otherwise

## Scheduler

| Key | Default | Description |
| --- | --- | --- |
| `cos_lr` | `false` | If `true`, uses cosine decay from `lr0` to `lr0 * lrf`; otherwise uses linear decay. |
| `warmup_epochs` | `3.0` | Warmup length in epochs. Converted to steps with `warmup_epochs * len(train_loader)`. |
| `warmup_momentum` | `0.8` | Momentum or beta1 value at the start of warmup. It linearly reaches `optimizer.momentum`. |

## Loss

Training always asks the adapter for `(total_loss, losses)`.

`loss.weights` controls whether the adapter's total loss is replaced by a
weighted sum of normalized or adapter-native loss terms:

```yaml
loss:
  weights:
    cls_loss: 1.0
    box_loss: 1.0
    obj_loss: 1.0
```

Behavior:

- If `loss.weights` is empty, `total_loss` from the adapter is used.
- If `loss.weights` contains universal keys such as `cls_loss`, `box_loss`,
  `obj_loss`, or `dfl_loss`, `train.py` maps them through `loss.aliases` to the
  adapter's native tensor losses, multiplies by the configured weights, and
  sums the result.
- If `loss.weights` contains adapter-native keys that are present in the
  adapter's `losses` dict, those tensor values are multiplied by their weights
  and summed.
- If no configured key matches a returned loss key, `total_loss` is used.

### Display Aliases

`loss.aliases` is loaded from `config/loss_aliases.yaml`. It is used both for
normalized display keys in `epoch_stats` and for resolving universal
`loss.weights` keys to adapter-native loss tensors.

```yaml
loss:
  aliases:
    cls_loss:
      - classification
      - cls_loss
      - class_loss
      - loss_cls
      - loss_class
      - loss_ce
      - loss_vfl
      - loss_labels
    box_loss:
      - bbox_regression
      - box_loss
      - iou_loss
      - giou_loss
      - l1_loss
      - loss_box
      - loss_iou
      - loss_giou
      - loss_l1
      - loss_bbox
    obj_loss:
      - obj_loss
      - objectness_loss
      - conf_loss
      - confidence_loss
      - loss_obj
      - loss_objectness
      - loss_conf
    dfl_loss:
      - dfl_loss
      - loss_dfl
```

For each alias, training sums any matching source keys present in the current
epoch's returned losses. Since `_print_epoch()` prints every float in
`epoch_stats`, aliases are displayed automatically when `train.verbose: true`.
If no alias source key matches, no normalized alias is added and the native
adapter loss keys are still displayed as returned.

## Presets

### RetinaNet

```yaml
# config/retinanet.yaml
model:
  name: retinanet
  variant: resnet50_fpn_v2
  weights: Default
  weights_backbone: null
  trainable_backbone_layers: 3

optimizer:
  name: SGD
  lr0: 0.005
  lrf: 0.01
  momentum: 0.9
  weight_decay: 0.0005

scheduler:
  cos_lr: false
  warmup_epochs: 1.0

loss:
  weights:
    cls_loss: 1.0
    box_loss: 1.0
```

### RT-DETR

```yaml
# config/rtdetr.yaml
model:
  name: rtdetr
  weights: Default
  score_threshold: 0.5

optimizer:
  name: AdamW
  lr0: 0.0001
  lrf: 0.01
  momentum: 0.9
  weight_decay: 0.0001

scheduler:
  cos_lr: true
  warmup_epochs: 1.0

loss:
  weights:
    cls_loss: 1.0
    box_loss: 1.0
```

### YOLOX

```yaml
# config/yolox.yaml
model:
  name: yolox
  variant: yolox-s
  weights: Default
  score_threshold: 0.3
  nms_threshold: 0.45

optimizer:
  name: SGD
  lr0: 0.01
  lrf: 0.01
  momentum: 0.9
  weight_decay: 0.0005
  nesterov: true

scheduler:
  cos_lr: true
  warmup_epochs: 5.0
  warmup_momentum: 0.8

loss:
  weights:
    cls_loss: 1.0
    box_loss: 1.0
    obj_loss: 1.0
```

## Defined But Not Used By Training

These keys exist in `config/default.yaml` but are not read by `train.py` in the
current implementation:

| Key | Notes |
| --- | --- |
| `augment.*` | No augmentation config is currently used. Training only resizes images to `train.imgsz`. |
