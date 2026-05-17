import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw
import torch

from config import load_config, load_model_config
import registry
from train import _apply_loss_weights, train_from_config


class TrainValidationSmokeTests(unittest.TestCase):
    def test_default_config_training_adds_validation_loss_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")

            result = train_from_config(
                None,
                data={"path": str(data_yaml)},
                model={"weights": None, "weights_backbone": None},
                train={
                    "epochs": 1,
                    "batch": 1,
                    "imgsz": 64,
                    "workers": 0,
                    "device": "cpu",
                    "project": str(tmp_path / "runs"),
                    "name": "default_val_smoke",
                    "exist_ok": True,
                    "save": False,
                    "verbose": False,
                    "amp": False,
                    "deterministic": False,
                    "val": True,
                },
            )

            stats = result["history"][0]
            self.assertEqual(result["config"]["model"]["name"], "retinanet")
            self.assertIn("cls_loss", stats)
            self.assertIn("box_loss", stats)
            self.assertIn("val_cls_loss", stats)
            self.assertIn("val_box_loss", stats)

    def test_retinanet_training_prints_normalized_train_and_val_losses(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")
            stdout = io.StringIO()

            with contextlib.redirect_stdout(stdout):
                result = train_from_config(
                    "config/retinanet.yaml",
                    data={"path": str(data_yaml)},
                    model={
                        "weights": None,
                        "weights_backbone": None,
                        "trainable_backbone_layers": 0,
                    },
                    train={
                        "epochs": 1,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "retinanet_val_smoke",
                        "exist_ok": True,
                        "save": False,
                        "verbose": True,
                        "amp": False,
                        "deterministic": False,
                        "val": True,
                    },
                )

            self.assertEqual(len(result["history"]), 1)
            stats = result["history"][0]
            for key in (
                "loss",
                "cls_loss",
                "box_loss",
                "val_loss",
                "val_cls_loss",
                "val_box_loss",
            ):
                self.assertIn(key, stats)
                self.assertIsInstance(stats[key], float)

            self.assertEqual(stats["cls_loss"], stats["classification"])
            self.assertEqual(stats["box_loss"], stats["bbox_regression"])
            self.assertEqual(stats["val_cls_loss"], stats["val_classification"])
            self.assertEqual(stats["val_box_loss"], stats["val_bbox_regression"])

            printed = stdout.getvalue()
            self.assertIn("epoch=1", printed)
            self.assertIn("loss=", printed)
            self.assertIn("cls_loss=", printed)
            self.assertIn("box_loss=", printed)
            self.assertIn("val_loss=", printed)
            self.assertIn("val_cls_loss=", printed)
            self.assertIn("val_box_loss=", printed)
            self.assertNotIn("classification=", printed)
            self.assertNotIn("bbox_regression=", printed)

    def test_yolox_training_adds_validation_loss_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")

            result = train_from_config(
                "config/yolox.yaml",
                data={"path": str(data_yaml)},
                model={"weights": None, "variant": "yolox-nano"},
                train={
                    "epochs": 1,
                    "batch": 1,
                    "imgsz": 64,
                    "workers": 0,
                    "device": "cpu",
                    "project": str(tmp_path / "runs"),
                    "name": "yolox_val_smoke",
                    "exist_ok": True,
                    "save": False,
                    "verbose": False,
                    "amp": False,
                    "deterministic": False,
                    "val": True,
                },
            )

            stats = result["history"][0]
            self.assertIn("cls_loss", stats)
            self.assertIn("box_loss", stats)
            self.assertIn("val_cls_loss", stats)
            self.assertIn("val_box_loss", stats)
            self.assertIn("val_iou_loss", stats)
            self.assertEqual(stats["box_loss"], stats["iou_loss"])
            self.assertEqual(stats["val_box_loss"], stats["val_iou_loss"])

    def test_rtdetr_training_adds_validation_loss_aliases(self):
        try:
            import transformers  # noqa: F401
        except ImportError:
            self.skipTest("RT-DETR requires transformers")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset", image_size=256)

            result = train_from_config(
                "config/rtdetr.yaml",
                data={"path": str(data_yaml)},
                model={"weights": None},
                train={
                    "epochs": 1,
                    "batch": 1,
                    "imgsz": 256,
                    "workers": 0,
                    "device": "cpu",
                    "project": str(tmp_path / "runs"),
                    "name": "rtdetr_val_smoke",
                    "exist_ok": True,
                    "save": False,
                    "verbose": False,
                    "amp": False,
                    "deterministic": False,
                    "val": True,
                },
            )

            stats = result["history"][0]
            self.assertIn("loss_vfl", stats)
            self.assertIn("loss_bbox", stats)
            self.assertIn("loss_giou", stats)
            self.assertIn("cls_loss", stats)
            self.assertIn("box_loss", stats)
            self.assertIn("val_loss_vfl", stats)
            self.assertIn("val_loss_bbox", stats)
            self.assertIn("val_loss_giou", stats)
            self.assertIn("val_cls_loss", stats)
            self.assertIn("val_box_loss", stats)
            self.assertEqual(stats["cls_loss"], stats["loss_vfl"])
            self.assertEqual(stats["box_loss"], stats["loss_bbox"] + stats["loss_giou"])
            self.assertEqual(stats["val_cls_loss"], stats["val_loss_vfl"])
            self.assertEqual(
                stats["val_box_loss"],
                stats["val_loss_bbox"] + stats["val_loss_giou"],
            )

    def test_training_can_disable_validation_from_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")

            result = train_from_config(
                "config/retinanet.yaml",
                data={"path": str(data_yaml)},
                model={
                    "weights": None,
                    "weights_backbone": None,
                    "trainable_backbone_layers": 0,
                },
                train={
                    "epochs": 1,
                    "batch": 1,
                    "imgsz": 64,
                    "workers": 0,
                    "device": "cpu",
                    "project": str(tmp_path / "runs"),
                    "name": "no_val_smoke",
                    "exist_ok": True,
                    "save": False,
                    "verbose": False,
                    "amp": False,
                    "deterministic": False,
                    "val": False,
                },
            )

            stats = result["history"][0]
            self.assertIn("cls_loss", stats)
            self.assertIn("box_loss", stats)
            self.assertNotIn("val_loss", stats)
            self.assertNotIn("val_cls_loss", stats)
            self.assertNotIn("val_box_loss", stats)

    def test_patience_stops_after_validation_loss_stops_improving(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")
            fake_builder = _make_fake_builder([1.0, 0.8, 0.9, 1.1])

            with _temporary_model("fake_patience", fake_builder):
                result = train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_patience"},
                    train={
                        "epochs": 10,
                        "patience": 2,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "patience_smoke",
                        "exist_ok": True,
                        "save": False,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": True,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

            self.assertEqual([row["epoch"] for row in result["history"]], [1, 2, 3, 4])
            self.assertEqual(
                [round(row["val_loss"], 1) for row in result["history"]],
                [1.0, 0.8, 0.9, 1.1],
            )

    def test_patience_is_ignored_when_validation_is_disabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")
            fake_builder = _make_fake_builder([1.0, 2.0, 3.0])

            with _temporary_model("fake_no_val_patience", fake_builder):
                result = train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_no_val_patience"},
                    train={
                        "epochs": 3,
                        "patience": 0,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "no_val_patience_smoke",
                        "exist_ok": True,
                        "save": False,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": False,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

            self.assertEqual(len(result["history"]), 3)
            self.assertNotIn("val_loss", result["history"][0])

    def test_resume_from_explicit_checkpoint_path_continues_epoch_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")
            run_dir = tmp_path / "runs" / "resume_explicit"
            first_builder = _make_fake_builder([1.0, 0.9], adapter_name="fake_resume_explicit")

            with _temporary_model("fake_resume_explicit", first_builder):
                first = train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_resume_explicit"},
                    train={
                        "epochs": 2,
                        "patience": 100,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "resume_explicit",
                        "exist_ok": True,
                        "save": True,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": True,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

            checkpoint_path = run_dir / "weights" / "last.pt"
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(checkpoint["epoch"], 1)
            self.assertEqual(checkpoint["adapter"], "fake_resume_explicit")
            self.assertAlmostEqual(checkpoint["best_val_loss"], 0.9, places=5)
            self.assertEqual(checkpoint["epochs_without_val_improvement"], 0)

            second_builder = _make_fake_builder([0.8, 0.7], adapter_name="fake_resume_explicit")
            with _temporary_model("fake_resume_explicit", second_builder):
                second = train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_resume_explicit"},
                    train={
                        "epochs": 4,
                        "patience": 100,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "resume_explicit",
                        "exist_ok": True,
                        "save": True,
                        "resume": str(checkpoint_path),
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": True,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

            self.assertEqual([row["epoch"] for row in first["history"]], [1, 2])
            self.assertEqual([row["epoch"] for row in second["history"]], [3, 4])
            resumed_checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.assertEqual(resumed_checkpoint["epoch"], 3)
            self.assertAlmostEqual(resumed_checkpoint["best_val_loss"], 0.7, places=5)

    def test_resume_true_uses_last_checkpoint_in_run_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")
            builder = _make_fake_builder([1.0, 0.9, 0.8], adapter_name="fake_resume_true")

            with _temporary_model("fake_resume_true", builder):
                train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_resume_true"},
                    train={
                        "epochs": 1,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "resume_true",
                        "exist_ok": True,
                        "save": True,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": True,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

            with _temporary_model("fake_resume_true", _make_fake_builder([0.7], adapter_name="fake_resume_true")):
                resumed = train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_resume_true"},
                    train={
                        "epochs": 2,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "resume_true",
                        "exist_ok": True,
                        "save": True,
                        "resume": True,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": True,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

            self.assertEqual([row["epoch"] for row in resumed["history"]], [2])

    def test_resume_rejects_checkpoint_from_different_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            data_yaml = _make_yolo_dataset(tmp_path / "dataset")

            with _temporary_model(
                "fake_resume_source",
                _make_fake_builder([1.0], adapter_name="fake_resume_source"),
            ):
                train_from_config(
                    None,
                    data={"path": str(data_yaml)},
                    model={"name": "fake_resume_source"},
                    train={
                        "epochs": 1,
                        "batch": 1,
                        "imgsz": 64,
                        "workers": 0,
                        "device": "cpu",
                        "project": str(tmp_path / "runs"),
                        "name": "adapter_mismatch",
                        "exist_ok": True,
                        "save": True,
                        "verbose": False,
                        "amp": False,
                        "deterministic": False,
                        "val": True,
                    },
                    optimizer={"name": "SGD", "lr0": 0.01},
                    scheduler={"warmup_epochs": 0.0},
                    loss={"weights": {}},
                )

            checkpoint_path = tmp_path / "runs" / "adapter_mismatch" / "weights" / "last.pt"
            with _temporary_model(
                "fake_resume_target",
                _make_fake_builder([1.0], adapter_name="fake_resume_target"),
            ):
                with self.assertRaisesRegex(ValueError, "Checkpoint adapter"):
                    train_from_config(
                        None,
                        data={"path": str(data_yaml)},
                        model={"name": "fake_resume_target"},
                        train={
                            "epochs": 2,
                            "batch": 1,
                            "imgsz": 64,
                            "workers": 0,
                            "device": "cpu",
                            "project": str(tmp_path / "runs"),
                            "name": "adapter_mismatch",
                            "exist_ok": True,
                            "save": False,
                            "resume": str(checkpoint_path),
                            "verbose": False,
                            "amp": False,
                            "deterministic": False,
                            "val": True,
                        },
                        optimizer={"name": "SGD", "lr0": 0.01},
                        scheduler={"warmup_epochs": 0.0},
                        loss={"weights": {}},
                    )


class ConfigLoadingTests(unittest.TestCase):
    def test_default_config_includes_loss_aliases(self):
        cfg = load_config()

        self.assertEqual(cfg["model"]["name"], "retinanet")
        self.assertIn("aliases", cfg["loss"])
        self.assertIn("cls_loss", cfg["loss"]["aliases"])
        self.assertIn("box_loss", cfg["loss"]["aliases"])
        self.assertIn("classification", cfg["loss"]["aliases"]["cls_loss"])
        self.assertIn("bbox_regression", cfg["loss"]["aliases"]["box_loss"])
        self.assertIn("loss_vfl", cfg["loss"]["aliases"]["cls_loss"])
        self.assertIn("loss_bbox", cfg["loss"]["aliases"]["box_loss"])
        self.assertIn("loss_giou", cfg["loss"]["aliases"]["box_loss"])
        self.assertIn("conf_loss", cfg["loss"]["aliases"]["obj_loss"])
        self.assertIn("loss_dfl", cfg["loss"]["aliases"]["dfl_loss"])

    def test_model_config_presets_merge_with_defaults_and_aliases(self):
        expected = {
            "retinanet": {
                "model_name": "retinanet",
                "optimizer": "SGD",
                "loss_keys": {"cls_loss", "box_loss"},
            },
            "yolox": {
                "model_name": "yolox",
                "optimizer": "SGD",
                "loss_keys": {"cls_loss", "box_loss", "obj_loss"},
            },
            "rtdetr": {
                "model_name": "rtdetr",
                "optimizer": "AdamW",
                "loss_keys": {"cls_loss", "box_loss"},
            },
        }

        for preset, checks in expected.items():
            with self.subTest(preset=preset):
                cfg = load_model_config(preset)
                self.assertEqual(cfg["model"]["name"], checks["model_name"])
                self.assertEqual(cfg["optimizer"]["name"], checks["optimizer"])
                self.assertEqual(set(cfg["loss"]["weights"]), checks["loss_keys"])
                self.assertIn("aliases", cfg["loss"])
                self.assertIn("cls_loss", cfg["loss"]["aliases"])
                self.assertIn("box_loss", cfg["loss"]["aliases"])
                self.assertEqual(cfg["data"]["train"], "train")
                self.assertEqual(cfg["data"]["val"], "val")
                self.assertTrue(cfg["train"]["val"])

    def test_overrides_deep_merge_without_dropping_aliases(self):
        cfg = load_config(
            "config/yolox.yaml",
            overrides={
                "data": {"path": "dataset/data.yaml"},
                "model": {"variant": "yolox-nano", "weights": None},
                "train": {"epochs": 1, "val": False},
                "loss": {"weights": {"cls_loss": 2.0}},
            },
        )

        self.assertEqual(cfg["data"]["path"], "dataset/data.yaml")
        self.assertEqual(cfg["data"]["train"], "train")
        self.assertEqual(cfg["model"]["name"], "yolox")
        self.assertEqual(cfg["model"]["variant"], "yolox-nano")
        self.assertIsNone(cfg["model"]["weights"])
        self.assertEqual(cfg["train"]["epochs"], 1)
        self.assertFalse(cfg["train"]["val"])
        self.assertEqual(cfg["loss"]["weights"]["cls_loss"], 2.0)
        self.assertEqual(cfg["loss"]["weights"]["box_loss"], 1.0)
        self.assertEqual(cfg["loss"]["weights"]["obj_loss"], 1.0)
        self.assertIn("aliases", cfg["loss"])
        self.assertIn("box_loss", cfg["loss"]["aliases"])


class LossWeightingTests(unittest.TestCase):
    def test_universal_loss_weights_map_to_native_loss_tensors(self):
        losses = {
            "classification": torch.tensor(2.0),
            "bbox_regression": torch.tensor(3.0),
        }
        aliases = {
            "cls_loss": ["classification"],
            "box_loss": ["bbox_regression"],
        }

        loss = _apply_loss_weights(
            total_loss=torch.tensor(100.0),
            losses=losses,
            loss_weights={"cls_loss": 0.5, "box_loss": 2.0},
            loss_aliases=aliases,
        )

        self.assertEqual(float(loss), 7.0)

    def test_universal_loss_weights_sum_multiple_native_terms(self):
        losses = {
            "iou_loss": torch.tensor(3.0),
            "l1_loss": torch.tensor(4.0),
            "conf_loss": torch.tensor(5.0),
        }
        aliases = {
            "box_loss": ["iou_loss", "l1_loss"],
            "obj_loss": ["conf_loss"],
        }

        loss = _apply_loss_weights(
            total_loss=torch.tensor(100.0),
            losses=losses,
            loss_weights={"box_loss": 1.0, "obj_loss": 0.1},
            loss_aliases=aliases,
        )

        self.assertEqual(float(loss), 7.5)

    def test_unmatched_loss_weights_fall_back_to_total_loss(self):
        loss = _apply_loss_weights(
            total_loss=torch.tensor(100.0),
            losses={"native_loss": torch.tensor(1.0)},
            loss_weights={"cls_loss": 1.0},
            loss_aliases={"cls_loss": ["missing_loss"]},
        )

        self.assertEqual(float(loss), 100.0)


def _make_yolo_dataset(root, image_size=96):
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True)

    train_images = _write_split(root, "train", image_count=2, image_size=image_size)
    val_images = _write_split(root, "val", image_count=1, image_size=image_size)

    train_txt = splits_dir / "train.txt"
    val_txt = splits_dir / "val.txt"
    train_txt.write_text("\n".join(train_images) + "\n")
    val_txt.write_text("\n".join(val_images) + "\n")

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {root}",
                "train: splits/train.txt",
                "val: splits/val.txt",
                "names:",
                "  0: object",
                "",
            ]
        )
    )
    return data_yaml


def _write_split(root, split, image_count, image_size):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    image_paths = []
    box_start = max(image_size // 4, 1)
    box_end = max(image_size - box_start, box_start + 1)
    for index in range(image_count):
        image_name = f"{split}_{index}.jpg"
        image_path = image_dir / image_name
        label_path = label_dir / f"{split}_{index}.txt"

        image = Image.new(
            "RGB",
            (image_size, image_size),
            (32 + index * 20, 48 + index * 12, 64 + index * 8),
        )
        draw = ImageDraw.Draw(image)
        draw.rectangle(
            (box_start, box_start, box_end, box_end),
            outline=(220, 80, 40),
            width=2,
        )
        image.save(image_path)
        label_path.write_text("0 0.5 0.5 0.5 0.5\n")
        image_paths.append(str(image_path.relative_to(root)))

    return image_paths


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


def _make_fake_builder(val_losses, adapter_name="fake"):
    state = {"val_index": 0}

    def build_fake_model(num_classes, **kwargs):
        return _FakeAdapter(val_losses=val_losses, state=state, name=adapter_name)

    return build_fake_model


class _FakeAdapter:
    def __init__(self, val_losses, state, name="fake"):
        self.name = name
        self.model = _FakeModel()
        self.val_losses = val_losses
        self.state = state

    def to(self, device):
        self.model.to(device)
        return self

    def training_step(self, images, targets):
        if not torch.is_grad_enabled():
            index = min(self.state["val_index"], len(self.val_losses) - 1)
            loss_value = self.val_losses[index]
            self.state["val_index"] += 1
        else:
            loss_value = 0.5

        loss = self.model.scale * 0.0 + torch.tensor(float(loss_value), device=self.model.scale.device)
        losses = {
            "classification": loss * 0.25,
            "bbox_regression": loss * 0.75,
        }
        return loss, losses


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))


if __name__ == "__main__":
    unittest.main()
