# Commands

Common commands for running Friendy Mercury experiments.

## Install Dependencies

Install the base dependencies:

```bash
pip install -r requirements.txt
```

Install RT-DETR support:

```bash
pip install -r requirements-rtdetr.txt
```

YOLOX model code is vendored in `vendor/yolox`, so no external YOLOX package is needed.



## Config Reference

See every supported YAML field and common model-specific options:

```bash
configs/config_reference.yaml
```

Description: this is a commented reference file. Use it as documentation and copy the fields you need into `configs/experiment.yaml`.

## Run Full Pipeline

Run the complete experiment from one config file:

```bash
python run.py configs/experiment.yaml
```

Description: loads the config, builds every dataset/model `ExperimentRun`, trains all runs, evaluates configured validation and test datasets from saved checkpoints, computes metrics, and exports CSV tables.

This command writes:

```text
runs/helmet-benchmark/results.yaml
runs/helmet-benchmark/val_results.yaml
runs/helmet-benchmark/test_results.yaml
runs/helmet-benchmark/metrics.csv
runs/helmet-benchmark/run_summary.yaml
```

Checkpoint choice is automatic: it uses `best.pt` when a validation dataset exists, otherwise `last.pt`.

## Check The Experiment Matrix

Print how many dataset/model runs the config expands into:

```bash
python -c "from config import load_config, build_experiment_runs; c = load_config('configs/experiment.yaml'); runs = build_experiment_runs(c); print(len(runs)); print([r.name for r in runs])"
```

Description: parses the YAML config and shows the generated `ExperimentRun` list. If you have 2 train datasets and 3 models, this should print 6 runs.

## Train All Runs

Train every dataset/model pair from the config:

```bash
python train.py configs/experiment.yaml
```

Description: loads the config, builds the `ExperimentRun` matrix, trains each run, saves checkpoints, writes histories/results, and runs test evaluation if a test dataset is configured.

Outputs are written under the configured `output_dir`, for example:

```text
runs/helmet-benchmark/
  config.resolved.yaml
  results.yaml
  00-field-v1-00-retinanet-resnet50_fpn_v2/
    last.pt
    best.pt
    history.yaml
    result.yaml
    test_predictions.pt
```

## Evaluate Trained Runs On Test

Evaluate saved checkpoints on the configured test dataset:

```bash
python val.py configs/experiment.yaml --split test --checkpoint best
```

Description: rebuilds every `ExperimentRun`, loads each run's `best.pt`, predicts on `run.test_dataset`, calls `evaluate_detection(...)`, and writes test metrics.

Use the last checkpoint instead:

```bash
python val.py configs/experiment.yaml --split test --checkpoint last
```

## Evaluate Trained Runs On Validation

Evaluate saved checkpoints on the configured validation dataset:

```bash
python val.py configs/experiment.yaml --split val --checkpoint best
```

Description: same evaluator as test, but uses `run.val_dataset` and writes validation results.

## Important Outputs

Top-level training results:

```text
runs/helmet-benchmark/results.yaml
```

Top-level validation results:

```text
runs/helmet-benchmark/val_results.yaml
```

Top-level test results:

```text
runs/helmet-benchmark/test_results.yaml
```

Universal run-level metrics table:

```text
runs/helmet-benchmark/metrics.csv
```

Per-run files:

```text
history.yaml
result.yaml
val_result.yaml
test_result.yaml
val_predictions.pt
test_predictions.pt
best.pt
last.pt
```


## Export Metrics To CSV

Export one universal CSV where each train dataset is one row and model results are grouped into model-prefixed columns:

```bash
python export.py runs/helmet-benchmark/results.yaml \
  --val-results runs/helmet-benchmark/val_results.yaml \
  --test-results runs/helmet-benchmark/test_results.yaml \
  --output runs/helmet-benchmark/metrics.csv
```

Description: merges training results, validation metrics, and test metrics into one matrix-friendly CSV. Columns include train/eval dataset identity plus model-prefixed run fields, training losses, `val_*` metrics, and `test_*` metrics such as `yolox_val_map50` or `retinanet_test_f1`.

If `--output` is omitted, the CSV is written next to the first YAML file with a `.csv` suffix.

## Metrics

Metrics are computed by:

```python
from metrics import evaluate_detection
```

The evaluator consumes Friendy-format predictions plus target dictionaries and returns:

```text
map50
map50_95
precision
recall
f1
num_images
num_predictions
num_targets
per_class
```

Per-class metrics include:

```text
ap50
ap50_95
precision
recall
f1
ground_truth_count
prediction_count
```

## Syntax Check

Check the main Python files without training:

```bash
python -m py_compile config.py data.py metrics.py train.py val.py registry.py formats.py
```

Description: catches syntax errors only. It does not validate datasets, checkpoints, or model dependencies.
