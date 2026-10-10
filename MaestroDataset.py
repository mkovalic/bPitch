"""Manifest-based MAESTRO windows with versioned, atomic feature caching."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import zipfile
import numpy as np
import pretty_midi
import torch
from torch.utils.data import Dataset
from audio_features import FEATURES, compute_hcqt, labels_for_notes, load_window


class MaestroDataset(Dataset):
    def __init__(
        self,
        data_dir,
        manifest="manifests/demo.json",
        split="train",
        cache_dir=".cache/features-v2",
    ):
        if split not in ("train", "validation", "test"):
            raise ValueError(f"Unknown split: {split}")
        self.data_dir = Path(data_dir).resolve()
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        if self.manifest.get("features") != FEATURES.to_dict():
            raise ValueError(
                "Manifest feature version differs; regenerate with prepare_data.py"
            )
        self.split = split
        self.cache_dir = Path(cache_dir).resolve() if cache_dir else None
        self.windows = [
            (track, start)
            for track in self.manifest["splits"][split]
            for start in track["start_samples"]
        ]
        if not self.windows:
            raise ValueError(f"Empty {split} dataset")
        self._notes = {}
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self):
        return len(self.windows)

    def cache_path(self, idx):
        track, start = self.windows[idx]
        paths = [self.data_dir / track[k] for k in ("audio_filename", "midi_filename")]
        identity = {
            "features": FEATURES.to_dict(),
            "start_sample": start,
            "files": [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths],
        }
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{key}.npz"

    def _read(self, idx):
        track, start = self.windows[idx]
        audio, mask = load_window(self.data_dir / track["audio_filename"], start)
        midi_path = str(self.data_dir / track["midi_filename"])
        if midi_path not in self._notes:
            midi = pretty_midi.PrettyMIDI(midi_path)
            self._notes[midi_path] = [
                n for inst in midi.instruments if not inst.is_drum for n in inst.notes
            ]
        return {
            "audio": compute_hcqt(audio),
            **labels_for_notes(self._notes[midi_path], start, mask),
        }

    def __getitem__(self, idx):
        cache = self.cache_path(idx) if self.cache_dir else None
        arrays = None
        if cache and cache.exists():
            try:
                with np.load(cache, allow_pickle=False) as stored:
                    arrays = {
                        key: stored[key]
                        for key in ("audio", "onset", "pitch", "note", "mask")
                    }
                self._validate(arrays)
            except (OSError, ValueError, KeyError, zipfile.BadZipFile):
                arrays = None
        if arrays is None:
            arrays = self._read(idx)
            arrays["audio"] = arrays["audio"].astype(np.float16)
            self._validate(arrays)
            if cache:
                with tempfile.NamedTemporaryFile(
                    dir=self.cache_dir, suffix=".npz", delete=False
                ) as tmp:
                    temporary = Path(tmp.name)
                try:
                    np.savez_compressed(
                        temporary,
                        audio=arrays["audio"].astype(np.float16),
                        **{
                            k: v.astype(np.uint8)
                            for k, v in arrays.items()
                            if k != "audio"
                        },
                    )
                    os.replace(temporary, cache)
                finally:
                    temporary.unlink(missing_ok=True)
        audio = torch.from_numpy(arrays["audio"].astype(np.float32))
        labels = {
            k: torch.from_numpy(v.astype(bool if k == "mask" else np.float32))
            for k, v in arrays.items()
            if k != "audio"
        }
        return audio, labels

    @staticmethod
    def _validate(arrays):
        expected = {
            "audio": (8, FEATURES.n_frames, FEATURES.n_pitch_bins),
            "pitch": (FEATURES.n_frames, FEATURES.n_pitch_bins),
            "onset": (FEATURES.n_frames, FEATURES.n_notes),
            "note": (FEATURES.n_frames, FEATURES.n_notes),
            "mask": (FEATURES.n_frames,),
        }
        for key, shape in expected.items():
            if arrays[key].shape != shape or not np.isfinite(arrays[key]).all():
                raise ValueError(f"Invalid {key} feature/label array")
