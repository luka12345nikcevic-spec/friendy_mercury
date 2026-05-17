import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
import torch

import registry
from train import train_from_config
from val import compute_detection_metrics, val_from_config


class ValSmokeTests(unittest.TestCase):
    def test_val_from_config_loads_best_checkpoint_and_scores_test_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset", include_test=True)

            with _temporary_model("fake_val", _build_fake_detector):
                train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_val"},
                    train={
                        "epochs": 1,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "fake_eval",
                        "exist_ok": True,
                        "save": True,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": False,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

                result = val_from_config(
                    None,
                    data={"path": str(data_yaml), "test": "test"},
                    model={"name": "fake_val"},
                    train={
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "fake_eval",
                    },
                )

            self.assertEqual(result["checkpoint"], tmp_path / "runs" / "fake_eval" / "weights" / "best.pt")
            self.assertEqual(result["metrics_path"], tmp_path / "runs" / "fake_eval" / "val_metrics.csv")
            metrics = result.metrics
            self.assertEqual(metrics["num_images"], 2)
            self.assertEqual(metrics["num_targets"], 2)
            self.assertEqual(metrics["num_predictions"], 2)
            self.assertEqual(metrics["precision"], 1.0)
            self.assertEqual(metrics["recall"], 1.0)
            self.assertEqual(metrics["map50"], 1.0)
            self.assertEqual(metrics["map50_95"], 1.0)
            self.assertEqual(metrics["per_class"][0]["precision"], 1.0)
            self.assertIn("speed", metrics)
            self.assertGreaterEqual(metrics["speed"]["inference_time_s"], 0.0)
            self.assertGreaterEqual(metrics["speed"]["inference_ms_per_image"], 0.0)
            self.assertGreaterEqual(metrics["speed"]["images_per_second"], 0.0)

            with open(result["metrics_path"], newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["checkpoint"], str(result["checkpoint"]))
            self.assertEqual(rows[0]["model"], "fake_val")
            self.assertEqual(float(rows[0]["precision"]), 1.0)
            self.assertEqual(float(rows[0]["recall"]), 1.0)
            self.assertEqual(float(rows[0]["map50"]), 1.0)
            self.assertEqual(float(rows[0]["map50_95"]), 1.0)
            self.assertEqual(int(rows[0]["num_images"]), 2)

    def test_val_appends_metrics_csv_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset", include_test=True)

            with _temporary_model("fake_val_append", _make_fake_detector_builder("fake_val_append")):
                train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_val_append"},
                    train={
                        "epochs": 1,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "fake_eval_append",
                        "exist_ok": True,
                        "save": True,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": False,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

                for _ in range(2):
                    result = val_from_config(
                        None,
                        data={"path": str(data_yaml), "test": "test"},
                        model={"name": "fake_val_append"},
                        train={
                            "batch": 1,
                            "imgsz": 64,
                            "workers": 0,
                            "device": "cpu",
                            "project": str(tmp_path / "runs"),
                            "name": "fake_eval_append",
                        },
                    )

            with open(result["metrics_path"], newline="") as file:
                rows = list(csv.DictReader(file))

            self.assertEqual(len(rows), 2)

    def test_val_raises_when_checkpoint_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset", include_test=True)

            with _temporary_model("fake_missing_checkpoint", _build_fake_detector):
                with self.assertRaisesRegex(FileNotFoundError, "Resume checkpoint does not exist"):
                    val_from_config(
                        None,
                        data={"path": str(data_yaml), "test": "test"},
                        model={"name": "fake_missing_checkpoint"},
                        train={
                            "project": str(tmp_path / "runs"),
                            "name": "missing",
                            "device": "cpu",
                        },
                    )

    def test_val_raises_when_test_split_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset", include_test=False)

            with _temporary_model("fake_missing_test", _build_fake_detector):
                with self.assertRaisesRegex(ValueError, "Validation requires data.yaml split 'test'"):
                    val_from_config(
                        None,
                        data={"path": str(data_yaml), "test": "test"},
                        model={"name": "fake_missing_test"},
                        train={
                            "project": str(tmp_path / "runs"),
                            "name": "missing",
                            "device": "cpu",
                        },
                    )


class MetricTests(unittest.TestCase):
    def test_compute_detection_metrics_counts_false_positive(self):
        predictions = [
            torch.tensor(
                [
                    [0.25, 0.25, 0.75, 0.75, 0.9, 0.0],
                    [0.00, 0.00, 0.20, 0.20, 0.8, 0.0],
                ]
            )
        ]
        targets = [torch.tensor([[0.25, 0.25, 0.75, 0.75, 1.0, 0.0]])]

        metrics = compute_detection_metrics(
            predictions=predictions,
            targets=targets,
            num_classes=1,
            iou_threshold=0.7,
        )

        self.assertEqual(metrics["precision"], 0.5)
        self.assertEqual(metrics["recall"], 1.0)
        self.assertEqual(metrics["map50"], 1.0)


class _TemporaryModel:
    def __init__(self, name, builder):
        self.name = name
        self.builder = builder
        self.previous = None

    def __enter__(self):
        self.previous = registry.MODEL_REGISTRY.get(self.name)
        registry.MODEL_REGISTRY[self.name] = self.builder
        return self

    def __exit__(self, exc_type, exc, traceback):
        if self.previous is None:
            registry.MODEL_REGISTRY.pop(self.name, None)
        else:
            registry.MODEL_REGISTRY[self.name] = self.previous


def _temporary_model(name, builder):
    return _TemporaryModel(name, builder)


def _build_fake_detector(num_classes, **kwargs):
    return _FakeDetector()


def _make_fake_detector_builder(name):
    def build_fake_detector(num_classes, **kwargs):
        return _FakeDetector(name=name)

    return build_fake_detector


class _FakeDetector:
    def __init__(self, name="fake_val"):
        self.name = name
        self.model = _FakeModel()

    def to(self, device):
        self.model.to(device)
        return self

    def eval(self):
        self.model.eval()
        return self

    def training_step(self, images, targets):
        loss = self.model.scale * 0.0 + torch.tensor(1.0, device=self.model.scale.device)
        return loss, {
            "classification": loss * 0.5,
            "bbox_regression": loss * 0.5,
        }

    @torch.no_grad()
    def predict(self, images):
        return [
            image.new_tensor([[0.5, 0.5, 0.5, 0.5, 0.95, 0.0]])
            for image in images
        ]


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))


def _make_yolo_dataset(root, include_test):
    train_images = _write_split(root, "train", image_count=1)
    val_images = _write_split(root, "val", image_count=1)
    test_images = _write_split(root, "test", image_count=2) if include_test else []

    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    (splits_dir / "train.txt").write_text("\n".join(train_images) + "\n")
    (splits_dir / "val.txt").write_text("\n".join(val_images) + "\n")
    if include_test:
        (splits_dir / "test.txt").write_text("\n".join(test_images) + "\n")

    lines = [
        f"path: {root}",
        "train: splits/train.txt",
        "val: splits/val.txt",
    ]
    if include_test:
        lines.append("test: splits/test.txt")
    lines.extend(["names:", "  0: object", ""])

    data_yaml = root / "data.yaml"
    data_yaml.write_text("\n".join(lines))
    return data_yaml


def _write_split(root, split, image_count):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    for index in range(image_count):
        image_path = image_dir / f"{split}_{index}.jpg"
        label_path = label_dir / f"{split}_{index}.txt"
        image = Image.new("RGB", (96, 96), (32, 48, 64))
        draw = ImageDraw.Draw(image)
        draw.rectangle((24, 24, 72, 72), outline=(220, 80, 40), width=2)
        image.save(image_path)
        label_path.write_text("0 0.5 0.5 0.5 0.5\n")
        image_paths.append(str(image_path.relative_to(root)))

    return image_paths


if __name__ == "__main__":
    unittest.main()
