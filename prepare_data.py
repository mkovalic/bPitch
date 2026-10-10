"""Select disjoint recordings from official splits and reproducible windows."""

import argparse
import csv
import json
from pathlib import Path
import random
import soundfile as sf
from audio_features import FEATURES
from project_config import manifest_digest, read_config


def build_manifest(
    rows, root, counts, windows_per_track, seed=42, holdout_manifest=None
):
    root = Path(root)
    if (
        holdout_manifest is not None
        and holdout_manifest.get("features") != FEATURES.to_dict()
    ):
        raise ValueError("Frozen holdouts use a different feature configuration")
    rng = random.Random(seed)
    splits = {}
    seen_paths, compositions = set(), {}
    for split in ("train", "validation", "test"):
        candidates = sorted(
            (r for r in rows if r["split"] == split), key=lambda r: r["audio_filename"]
        )
        if (
            counts[split] <= 0
            or counts[split] > len(candidates)
            or windows_per_track[split] <= 0
        ):
            raise ValueError(f"Invalid track/window count for {split}")
        frozen = None
        if holdout_manifest is not None and split != "train":
            source = holdout_manifest["splits"][split]
            frozen = {track["audio_filename"]: track for track in source}
            if len(source) != counts[split] or len(frozen) != len(source):
                raise ValueError(f"Frozen {split} recording count differs from config")
            official = {row["audio_filename"]: row for row in candidates}
            selected = []
            for track in source:
                row = official.get(track["audio_filename"])
                if row is None or any(
                    row[key] != track[key]
                    for key in (
                        "midi_filename",
                        "canonical_composer",
                        "canonical_title",
                    )
                ):
                    raise ValueError(
                        f"Frozen recording does not match official {split} split"
                    )
                selected.append(row)
        else:
            rng.shuffle(candidates)
            selected = sorted(
                candidates[: counts[split]], key=lambda row: row["audio_filename"]
            )
        tracks = []
        for row in selected:
            audio, midi = root / row["audio_filename"], root / row["midi_filename"]
            if not audio.is_file() or not midi.is_file():
                raise FileNotFoundError(f"Missing pair: {audio}, {midi}")
            if row["audio_filename"] in seen_paths:
                raise ValueError("Recording appears in multiple splits")
            seen_paths.add(row["audio_filename"])
            composition = (row["canonical_composer"], row["canonical_title"])
            if composition in compositions and compositions[composition] != split:
                raise ValueError(
                    f"Composition appears in multiple splits: {composition}"
                )
            compositions[composition] = split
            duration = min(float(row["duration"]), sf.info(audio).duration)
            if duration <= 0:
                raise ValueError(f"Empty recording: {audio}")
            max_start = max(
                0, int(duration * FEATURES.sample_rate) - FEATURES.n_samples
            )
            n = windows_per_track[split]
            if frozen is not None:
                starts = list(frozen[row["audio_filename"]]["start_samples"])
                if len(starts) != n or any(
                    not isinstance(start, int)
                    or not 0 <= start <= max_start
                    or start % FEATURES.hop_length
                    for start in starts
                ):
                    raise ValueError(
                        f"Frozen {split} windows differ from config or audio bounds"
                    )
            else:
                fractions = [
                    (i + (rng.random() if split == "train" else 0.5)) / n
                    for i in range(n)
                ]
                starts = [
                    int(f * max_start // FEATURES.hop_length) * FEATURES.hop_length
                    for f in fractions
                ]
            tracks.append(
                {
                    k: row[k]
                    for k in (
                        "audio_filename",
                        "midi_filename",
                        "canonical_composer",
                        "canonical_title",
                    )
                }
            )
            tracks[-1].update(duration=duration, start_samples=starts)
        splits[split] = tracks
    return {
        "version": 2,
        "seed": seed,
        "features": FEATURES.to_dict(),
        "splits": splits,
    }


def prepare(config, overwrite=False):
    path = Path(config["manifest"])
    with open(Path(config["data_dir"]) / "maestro-v3.0.0.csv") as source:
        rows = list(csv.DictReader(source))
    holdout_path = config.get("holdout_manifest")
    manifest = build_manifest(
        rows,
        config["data_dir"],
        config["tracks"],
        config["windows_per_track"],
        config["seed"],
        json.loads(Path(holdout_path).read_text()) if holdout_path else None,
    )
    if holdout_path:
        manifest["holdout_source_sha256"] = manifest_digest(holdout_path)
    if path.exists() and json.loads(path.read_text()) != manifest and not overwrite:
        raise ValueError(
            "Manifest already exists with different settings. Use --overwrite intentionally."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--cache", action="store_true", help="Precompute all selected windows"
    )
    args = parser.parse_args()
    config = read_config(args.config)
    manifest = prepare(config, args.overwrite)
    for split, tracks in manifest["splits"].items():
        count = sum(len(t["start_samples"]) for t in tracks)
        print(f"{split}: {len(tracks)} recordings, {count} windows")
        if args.cache:
            from MaestroDataset import MaestroDataset

            dataset = MaestroDataset(
                config["data_dir"], config["manifest"], split, config["cache_dir"]
            )
            for index in range(len(dataset)):
                dataset[index]
                if (index + 1) % 50 == 0 or index + 1 == len(dataset):
                    print(f"Cached {split}: {index + 1}/{len(dataset)}", flush=True)
    print(f"Manifest: {config['manifest']}")


if __name__ == "__main__":
    main()
