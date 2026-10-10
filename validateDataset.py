"""Validate metadata pairs and audio headers without decoding 120 GB of audio."""

import argparse
import csv
from pathlib import Path
import soundfile as sf
from project_config import read_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_config(args.config)
    root = Path(config["data_dir"])
    rows = list(csv.DictReader((root / "maestro-v3.0.0.csv").open()))
    errors = []
    for row in rows:
        try:
            if not (root / row["midi_filename"]).is_file():
                raise FileNotFoundError(row["midi_filename"])
            info = sf.info(root / row["audio_filename"])
            if info.frames <= 0:
                raise ValueError("Empty audio")
        except (OSError, RuntimeError, ValueError) as error:
            errors.append(f"{row['audio_filename']}: {error}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Validated {len(rows)} audio/MIDI pairs and audio headers.")


if __name__ == "__main__":
    main()
