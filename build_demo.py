"""Build a static results viewer from frozen test metrics and held-out excerpts."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import warnings

import mir_eval
import numpy as np
import pretty_midi

from audio_features import FEATURES
from performance_metrics import counts_to_metrics
from project_config import load_checkpoint, manifest_digest


def select_clips(checkpoint, duration=8.0):
    """Choose musical textures using reference MIDI only, never model scores."""
    candidates = []
    tracks = checkpoint["manifest"]["splits"]["test"]
    for track_index, track in enumerate(tracks):
        midi = pretty_midi.PrettyMIDI(
            str(Path(checkpoint["config"]["data_dir"]) / track["midi_filename"])
        )
        notes = [
            n
            for instrument in midi.instruments
            if not instrument.is_drum
            for n in instrument.notes
            if FEATURES.min_midi <= n.pitch < FEATURES.min_midi + FEATURES.n_notes
        ]
        for window_index, start_sample in enumerate(track["start_samples"]):
            start = start_sample / FEATURES.sample_rate
            if float(track["duration"]) - start < duration:
                continue
            excerpt = [n for n in notes if n.end > start and n.start < start + duration]
            onsets = sorted(n.start for n in excerpt if n.start >= start)
            if len(onsets) < 8:
                continue
            groups = []
            for onset in onsets:
                if not groups or onset - groups[-1][0] > 0.03:
                    groups.append([onset, 1])
                else:
                    groups[-1][1] += 1
            candidates.append(
                {
                    "track_index": track_index,
                    "window_index": window_index,
                    "start": start,
                    "duration": duration,
                    "onsets_per_second": len(onsets) / duration,
                    "chord_fraction": sum(g[1] >= 3 for g in groups) / len(groups),
                    "mean_polyphony": sum(
                        min(n.end, start + duration) - max(n.start, start)
                        for n in excerpt
                    )
                    / duration,
                }
            )
    # Preserve the previously inspected example as an explicit limitation case.
    limitation = next(
        c for c in candidates if c["track_index"] == 0 and c["window_index"] == 0
    )
    selected, used = [], {0}
    choices = [
        (
            "sparse",
            "Sparse passage",
            "Listen for isolated attacks and the spaces between phrases.",
            lambda c: (c["onsets_per_second"], c["mean_polyphony"]),
        ),
        (
            "chords",
            "Chords",
            "Compare simultaneous notes with the stacked reference notes.",
            lambda c: (-c["chord_fraction"], -c["mean_polyphony"]),
        ),
        (
            "fast",
            "Fast playing",
            "Follow the rapid attacks and look for missed or extra notes.",
            lambda c: (-c["onsets_per_second"], -c["mean_polyphony"]),
        ),
    ]
    for slug, label, description, key in choices:
        available = [c for c in candidates if c["track_index"] not in used]
        if not available:
            raise ValueError("The demo needs four distinct held-out recordings")
        chosen = min(available, key=key)
        used.add(chosen["track_index"])
        selected.append(
            {**chosen, "id": slug, "label": label, "description": description}
        )
    selected.append(
        {
            **limitation,
            "id": "endings",
            "label": "Note-ending errors",
            "description": "A known limitation: attacks can match while note endings are too early or too late.",
        }
    )
    for clip in selected:
        track = tracks[clip["track_index"]]
        clip.update(
            composer=track["canonical_composer"], title=track["canonical_title"]
        )
    return selected


def note_diagnostics(predicted, reference):
    """Explain duration errors among pitch/onset matches, using evaluator tolerances."""
    if not predicted or not reference:
        return {
            "with_offset": counts_to_metrics(0, len(predicted), len(reference)),
            "onset_matches": 0,
            "ending_matches_among_onset_matches": 0,
            "extra_notes": len(predicted),
            "missed_notes": len(reference),
            "duration_examples": [],
        }

    def unpack(notes):
        return (
            np.array([[n[0], n[1]] for n in notes], dtype=float),
            np.array([pretty_midi.note_number_to_hz(n[2]) for n in notes]),
        )

    ref_intervals, ref_pitches = unpack(reference)
    est_intervals, est_pitches = unpack(predicted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        matches = mir_eval.transcription.match_notes(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50,
            offset_ratio=None,
        )
        with_offset = mir_eval.transcription.match_notes(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50,
            offset_ratio=0.2,
        )
    errors, ending_matches = [], 0
    for ref_index, est_index in matches:
        ref, est = reference[ref_index], predicted[est_index]
        delta = est[1] - ref[1]
        tolerance = max(0.05, 0.2 * (ref[1] - ref[0]))
        if abs(delta) <= tolerance:
            ending_matches += 1
        else:
            errors.append(
                {
                    "pitch": int(ref[2]),
                    "note": pretty_midi.note_number_to_name(ref[2]),
                    "reference_start": ref[0],
                    "predicted_start": est[0],
                    "reference_end": ref[1],
                    "predicted_end": est[1],
                    "end_error_seconds": delta,
                }
            )
    errors.sort(key=lambda e: abs(e["end_error_seconds"]), reverse=True)
    return {
        "with_offset": counts_to_metrics(
            len(with_offset), len(predicted), len(reference)
        ),
        "onset_matches": len(matches),
        "ending_matches_among_onset_matches": ending_matches,
        "extra_notes": len(predicted) - len(matches),
        "missed_notes": len(reference) - len(matches),
        "duration_examples": errors[:3],
    }


def build_demo(checkpoint_path, baseline_path, output):
    checkpoint_path, baseline_path, output = map(
        Path, (checkpoint_path, baseline_path, output)
    )
    checkpoint, baseline = (
        load_checkpoint(checkpoint_path),
        load_checkpoint(baseline_path),
    )
    for saved in (checkpoint, baseline):
        if saved["manifest_sha256"] != manifest_digest(saved["config"]["manifest"]):
            raise ValueError("Manifest changed after training")
    for split in ("validation", "test"):
        if (
            checkpoint["manifest"]["splits"][split]
            != baseline["manifest"]["splits"][split]
        ):
            raise ValueError(
                "Comparison requires identical validation and test windows"
            )
    current_metrics = json.loads(
        (checkpoint_path.parent / "test_metrics.json").read_text()
    )
    baseline_metrics = json.loads(
        (baseline_path.parent / "test_metrics.json").read_text()
    )
    decoder = json.loads((checkpoint_path.parent / "decoder.json").read_text())
    if Path(decoder["checkpoint"]).resolve() != checkpoint_path.resolve():
        raise ValueError("Decoder was calibrated for another checkpoint")
    if current_metrics["thresholds"] != decoder["thresholds"]:
        raise ValueError("Test metrics and calibrated decoder differ")
    expected_windows = sum(
        len(t["start_samples"]) for t in checkpoint["manifest"]["splits"]["test"]
    )
    if any(
        m["split"] != "test" or m["windows"] != expected_windows
        for m in (current_metrics, baseline_metrics)
    ):
        raise ValueError("Comparison requires complete held-out test metrics")
    output.mkdir(parents=True, exist_ok=True)
    clips = select_clips(checkpoint)
    for clip in clips:
        clip_dir = output / "clips" / clip["id"]
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).with_name("transcribe.py")),
                "--checkpoint",
                str(checkpoint_path.resolve()),
                "--example",
                "test",
                "--example-index",
                str(clip["track_index"]),
                "--start",
                str(clip["start"]),
                "--duration",
                str(clip["duration"]),
                "--output",
                str(clip_dir.resolve()),
            ],
            check=True,
        )
        events = json.loads((clip_dir / "events.json").read_text())
        clip["metrics"] = events["metrics"]
        clip["diagnostics"] = note_diagnostics(events["predicted"], events["reference"])
        clip["assets"] = {
            name: f"clips/{clip['id']}/{name}"
            for name in (
                "original.wav",
                "prediction.wav",
                "prediction.mid",
                "reference.mid",
                "piano_roll.png",
                "index.html",
            )
        }
        print(
            f"{clip['label']}: note F1 {clip['metrics']['f1']:.1%}; "
            f"with endings {clip['diagnostics']['with_offset']['f1']:.1%}",
            flush=True,
        )
    data = {
        "current": current_metrics,
        "baseline": baseline_metrics,
        "clips": clips,
        "checkpoint": str(checkpoint_path.resolve()),
        "best_epoch": checkpoint["epoch"] + 1,
        "training_recordings": len(checkpoint["manifest"]["splits"]["train"]),
        "baseline_training_recordings": len(baseline["manifest"]["splits"]["train"]),
        "test_recordings": len(checkpoint["manifest"]["splits"]["test"]),
        "validation_windows": decoder["validation_windows"],
        "selection": "Reference MIDI only; no prediction scores used to choose textures. "
        "The previously inspected test excerpt is retained to illustrate note-ending errors.",
    }
    (output / "results.json").write_text(json.dumps(data, indent=2) + "\n")
    template = Path(__file__).with_name("demo_viewer.html").read_text()
    payload = (
        json.dumps(data)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )
    (output / "index.html").write_text(template.replace("__DEMO_DATA__", payload))
    print(f"Results viewer: {output.resolve() / 'index.html'}", flush=True)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/expanded/best.pt")
    parser.add_argument("--baseline", default="runs/demo/best.pt")
    parser.add_argument("--output", default="runs/expanded/viewer")
    args = parser.parse_args()
    build_demo(args.checkpoint, args.baseline, args.output)


if __name__ == "__main__":
    main()
