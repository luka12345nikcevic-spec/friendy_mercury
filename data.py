from pathlib import Path

import yaml


IMAGE_EXTENSIONS = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def load_data_yaml(data_yaml_path):
    data_yaml_path = Path(data_yaml_path).resolve()

    with open(data_yaml_path) as file:
        config = yaml.safe_load(file) or {}

    dataset_root = _resolve_dataset_root(config.get("path"), data_yaml_path.parent)
    names = _normalize_names(config.get("names", {}))

    return {
        "yaml_path": data_yaml_path,
        "root": dataset_root,
        "names": names,
        "train": config.get("train"),
        "val": config.get("val"),
        "test": config.get("test"),
    }


def _resolve_dataset_root(config_path, yaml_parent):
    if not config_path:
        return yaml_parent

    candidate = Path(config_path)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    if not candidate.is_absolute():
        relative_candidate = (yaml_parent / candidate).resolve()
        if relative_candidate.exists():
            return relative_candidate

    return yaml_parent


def _normalize_names(names):
    if isinstance(names, list):
        return {index: name for index, name in enumerate(names)}

    return {
        int(class_id): class_name
        for class_id, class_name in names.items()
    }


def _resolve_split_entries(root, split_value):
    if split_value is None:
        raise ValueError("Requested split is not defined in data.yaml")

    if isinstance(split_value, (list, tuple)):
        image_paths = []
        for entry in split_value:
            image_paths.extend(_resolve_split_entries(root, entry))
        return sorted(image_paths)

    split_path = Path(split_value)
    if not split_path.is_absolute():
        split_path = root / split_path

    if split_path.is_file() and split_path.suffix.lower() == ".txt":
        image_paths = []
        with open(split_path) as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue

                image_path = Path(line)
                if not image_path.is_absolute():
                    image_path = root / image_path
                image_paths.append(image_path.resolve())

        return sorted(image_paths)

    if split_path.is_dir():
        return sorted(
            path.resolve()
            for path in split_path.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

    raise FileNotFoundError(f"Could not resolve split path: {split_path}")


def _image_to_label_path(image_path):
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def _xywhn_to_xyxy(box, image_width, image_height):
    x_center, y_center, width, height = box
    x1 = (x_center - width / 2) * image_width
    y1 = (y_center - height / 2) * image_height
    x2 = (x_center + width / 2) * image_width
    y2 = (y_center + height / 2) * image_height
    return [x1, y1, x2, y2]


def _read_yolo_label_file(label_path, image_width, image_height):
    boxes = []
    labels = []

    if not label_path.exists():
        return boxes, labels

    with open(label_path) as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(float(parts[0]))
            yolo_box = [float(value) for value in parts[1:5]]
            boxes.append(_xywhn_to_xyxy(yolo_box, image_width, image_height))
            labels.append(class_id)

    return boxes, labels


class YoloDetectionDataset:
    def __init__(self, data_yaml_path, split="train", transforms=None):
        self.data = load_data_yaml(data_yaml_path)
        self.split = split
        self.transforms = transforms
        self.image_paths = _resolve_split_entries(self.data["root"], self.data[split])
        self.names = self.data["names"]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        import numpy as np
        import torch
        from PIL import Image

        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        label_path = _image_to_label_path(image_path)
        boxes, labels = _read_yolo_label_file(label_path, width, height)

        image = np.asarray(image).copy()
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels_tensor = torch.tensor(labels, dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index], dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(labels),), dtype=torch.int64),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "orig_size": torch.tensor([height, width], dtype=torch.int64),
        }

        if self.transforms is not None:
            image_tensor, target = self.transforms(image_tensor, target)

        return image_tensor, target


def detection_collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
