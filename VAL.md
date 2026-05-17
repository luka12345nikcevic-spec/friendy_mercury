# Validation And Evaluation

`val.py` evaluates a saved training checkpoint on a YOLO-format test split and
returns comparable detection metrics.

```python
from val import val_from_config

result = val_from_config(
    "config/retinanet.yaml",
    data={"path": "path/to/data.yaml", "test": "test"},
)
print(result.metrics)
```

`val_from_config()` uses the same config loader as training: it starts from
`config/default.yaml`, deep-merges `config/loss_aliases.yaml`, deep-merges the
config file you pass, then deep-merges keyword overrides.

## Required Data

`data.path` must point to a YOLO `data.yaml`.

`val.py` uses `data.test` as the name of the split key inside that dataset YAML.
For example, with:

```yaml
data:
  path: /datasets/my_dataset/data.yaml
  test: test
```

the dataset YAML must contain:

```yaml
path: /datasets/my_dataset
test: images/test
names:
  0: class_a
  1: class_b
```

The selected test split may be a directory, a `.txt` file of image paths, or a
list of entries. Image labels are resolved by replacing an `images` path
component with `labels` and changing the suffix to `.txt`.

If the configured test split is missing, `val.py` raises a `ValueError`.

## Checkpoint Resolution

By default:

```yaml
eval:
  weights: best.pt
```

resolves to:

```text
{train.project}/{train.name}/weights/best.pt
```

With the default training run fields:

```yaml
train:
  project: runs/train
  name: exp
```

that means:

```text
runs/train/exp/weights/best.pt
```

You can also evaluate a different checkpoint from the same run:

```yaml
eval:
  weights: last.pt
```

or:

```yaml
eval:
  weights: epoch10.pt
```

You can also pass an explicit path:

```yaml
eval:
  weights: /absolute/path/to/checkpoint.pt
```

If the checkpoint is missing, `val.py` raises a `FileNotFoundError`.

## Evaluation Config

```yaml
eval:
  weights: best.pt
  conf: 0.001
  iou: 0.7
  max_det: 300
```

| Key | Default | Description |
| --- | --- | --- |
| `eval.weights` | `best.pt` | Checkpoint name under `{train.project}/{train.name}/weights`, or an explicit checkpoint path. |
| `eval.conf` | `0.001` | Minimum prediction confidence kept for metric evaluation. |
| `eval.iou` | `0.7` | IoU threshold used for reported precision and recall. |
| `eval.max_det` | `300` | Maximum predictions kept per image after confidence filtering. |

## Metrics

`val.py` returns a `ValResult` dict with:

```python
{
    "checkpoint": checkpoint_path,
    "metrics": metrics,
    "metrics_path": metrics_path,
    "config": cfg,
    "names": names,
}
```

`result.metrics` contains:

- `precision`
- `recall`
- `map50`
- `map50_95`
- `per_class`
- `num_images`
- `num_predictions`
- `num_targets`
- `speed`

`precision` and `recall` use `eval.iou`. `map50` uses IoU `0.50`.
`map50_95` averages AP across IoU thresholds `0.50, 0.55, ..., 0.95`.

`speed` contains model inference timing measured around `adapter.predict()`:

```python
{
    "inference_time_s": 1.23,
    "inference_ms_per_image": 12.3,
    "images_per_second": 81.3,
}
```

Image loading, resize transforms, target conversion, metric calculation, and
post-metric filtering are not included in the timing. CUDA runs are synchronized
around prediction calls before measuring elapsed time.

## Metrics CSV

Every `val_from_config()` call appends one row to:

```text
{train.project}/{train.name}/val_metrics.csv
```

For a default run, that is:

```text
runs/train/exp/val_metrics.csv
```

The CSV stores flattened summary metrics and speed fields:

- `checkpoint`
- `model`
- `eval_conf`
- `eval_iou`
- `eval_max_det`
- `precision`
- `recall`
- `map50`
- `map50_95`
- `num_images`
- `num_predictions`
- `num_targets`
- `inference_time_s`
- `inference_ms_per_image`
- `images_per_second`

Per-class metrics remain in `result.metrics["per_class"]`; they are not written
to the summary CSV.

## Model Loading

The checkpoint stores the adapter name and model state. `val.py` rebuilds the
model from the current config, verifies the checkpoint adapter matches
`model.name`, and then loads `checkpoint["model"]`.

For evaluation, `model.weights` is forced to `null` during rebuild because the
checkpoint model state is the source of truth.
