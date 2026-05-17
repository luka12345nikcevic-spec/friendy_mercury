from copy import deepcopy
from pathlib import Path

import yaml


CONFIG_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.yaml"
LOSS_ALIASES_CONFIG_PATH = CONFIG_DIR / "loss_aliases.yaml"


def load_config(config_path=None, overrides=None):
    config = _read_yaml(DEFAULT_CONFIG_PATH)
    config = deep_update(config, _read_yaml(LOSS_ALIASES_CONFIG_PATH))

    if config_path is not None:
        config = deep_update(config, _read_yaml(config_path))

    if overrides:
        config = deep_update(config, overrides)

    return config


def load_model_config(model_name, overrides=None):
    config_path = CONFIG_DIR / f"{model_name}.yaml"
    return load_config(config_path, overrides=overrides)


def deep_update(base, updates):
    result = deepcopy(base)
    for key, value in (updates or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def _read_yaml(path):
    path = Path(path)
    with open(path) as file:
        return yaml.safe_load(file) or {}
