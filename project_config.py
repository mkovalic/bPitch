"""Resolve config paths relative to the config file, independent of cwd."""

import hashlib
import json
from pathlib import Path
import random
import numpy as np
import torch

DEFAULTS = {
    "data_dir": "data/maestro-v3.0.0",
    "manifest": "manifests/demo.json",
    "cache_dir": ".cache/features-v2",
    "run_dir": "runs/demo",
    "seed": 42,
    "tracks": {"train": 50, "validation": 10, "test": 10},
    "windows_per_track": {"train": 32, "validation": 16, "test": 16},
    "batch_size": 4,
    "num_workers": 2,
    "epochs": 50,
    "learning_rate": 0.001,
    "patience": 8,
    "observability": {
        "enabled": True,
        "log_every_batches": 25,
        "validation_windows": 16,
        "media_every_epochs": 5,
        "decoder": {
            "threshold_onset": 0.995,
            "threshold_note": 0.3,
            "min_duration_ms": 40,
            "tolerance_frames": 11,
        },
    },
}


def read_config(path="config.json"):
    path = Path(path).resolve()
    supplied = json.loads(path.read_text())
    config = {**DEFAULTS, **supplied}
    for key in ("tracks", "windows_per_track"):
        config[key] = {**DEFAULTS[key], **supplied.get(key, {})}
    config["observability"] = {
        **DEFAULTS["observability"],
        **supplied.get("observability", {}),
    }
    config["observability"]["decoder"] = {
        **DEFAULTS["observability"]["decoder"],
        **supplied.get("observability", {}).get("decoder", {}),
    }
    for key in ("data_dir", "manifest", "cache_dir", "run_dir", "holdout_manifest"):
        if config.get(key):
            config[key] = str((path.parent / config[key]).resolve())
    return config


def manifest_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def choose_device(request="auto"):
    if request != "auto":
        return torch.device(request)
    return torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    from audio_features import FEATURES

    if (
        checkpoint.get("model_version") != 2
        or checkpoint.get("features") != FEATURES.to_dict()
    ):
        raise ValueError(
            "Legacy/incompatible checkpoint. Retrain the corrected version-2 pipeline."
        )
    return checkpoint
