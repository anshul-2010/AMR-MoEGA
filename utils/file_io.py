"""
Helpers for standardized file I/O (paths, saving/loading objects).
"""

import os
import json
import yaml
import pickle
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_yaml(path: str) -> Dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def save_json(data: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)
