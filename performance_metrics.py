"""Strict frame metrics and note-event evaluation; thresholds use validation only."""

import argparse
import json
from pathlib import Path
import warnings
import mir_eval
import numpy as np
import pretty_midi
import torch
from torch.utils.data import DataLoader
from audio_features import FEATURES
from MaestroDataset import MaestroDataset
from model import BasicPitchModel
from note_creator import NoteCreator
from project_config import choose_device, load_checkpoint, manifest_digest


def counts_to_metrics(tp, predicted, reference):
    tp, predicted, reference = int(tp), int(predicted), int(reference)
    precision = tp / predicted if predicted else 0.0
    recall = tp / reference if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matched": int(tp),
        "predicted": int(predicted),
        "reference": int(reference),
    }


def frame_counts(predictions, truth, mask=None, threshold=0.5):
    if predictions.shape != truth.shape:
        raise ValueError(
            f"Metric shapes must match exactly: {predictions.shape} vs {truth.shape}"
        )
    predicted, reference = predictions >= threshold, truth >= 0.5
    if mask is not None:
        valid = mask.unsqueeze(-1).expand_as(reference)
        predicted, reference = predicted & valid, reference & valid
    return (
        int((predicted & reference).sum()),
        int(predicted.sum()),
        int(reference.sum()),
    )


def calculate_metrics(predictions, ground_truths, threshold=0.5, mask=None):
    return counts_to_metrics(*frame_counts(predictions, ground_truths, mask, threshold))


def reference_notes(dataset, index):
    track, start = dataset.windows[index]
    path = str(dataset.data_dir / track["midi_filename"])
    if path not in dataset._notes:
        midi = pretty_midi.PrettyMIDI(path)
        dataset._notes[path] = [
            n for inst in midi.instruments if not inst.is_drum for n in inst.notes
        ]
    offset, end = (
        start / FEATURES.sample_rate,
        start / FEATURES.sample_rate + FEATURES.window_seconds,
    )
    return [
        (max(0.0, n.start - offset), min(end, n.end) - offset, n.pitch)
        for n in dataset._notes[path]
        if n.end > offset
        and n.start < end
        and FEATURES.min_midi <= n.pitch < FEATURES.min_midi + FEATURES.n_notes
    ]


def note_counts(predicted, reference, offsets=False):
    if not predicted or not reference:
        return 0, len(predicted), len(reference)

    def unpack(notes):
        intervals = np.array([[n[0], n[1]] for n in notes], dtype=float)
        pitches = np.array([pretty_midi.note_number_to_hz(n[2]) for n in notes])
        return intervals, pitches

    ref_intervals, ref_pitches = unpack(reference)
    est_intervals, est_pitches = unpack(predicted)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        matched = mir_eval.transcription.match_notes(
            ref_intervals,
            ref_pitches,
            est_intervals,
            est_pitches,
            onset_tolerance=0.05,
            pitch_tolerance=50.0,
            offset_ratio=0.2 if offsets else None,
        )
    return len(matched), len(predicted), len(reference)


def collect_predictions(model, dataset, device, batch_size=4, max_windows=None):
    result = []
    model.eval()
    with torch.inference_mode():
        for inputs, labels in DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        ):
            outputs = {k: v.cpu().numpy() for k, v in model(inputs.to(device)).items()}
            for b in range(len(inputs)):
                index = len(result)
                valid = labels["mask"][b].numpy()
                result.append(
                    {
                        "outputs": {k: v[b][valid] for k, v in outputs.items()},
                        "labels": {
                            k: v[b].numpy()[valid]
                            for k, v in labels.items()
                            if k != "mask"
                        },
                        "reference": reference_notes(dataset, index),
                    }
                )
                if max_windows and len(result) >= max_windows:
                    return result
    return result


def score_predictions(examples, thresholds, offsets=False):
    counts = np.zeros(3, dtype=np.int64)
    decoder = NoteCreator(**thresholds)
    for example in examples:
        outputs = example["outputs"]
        predicted, _ = decoder.create_notes(outputs["onset"], outputs["note"])
        counts += note_counts(predicted, example["reference"], offsets)
    return counts_to_metrics(*counts)


def tune_thresholds(examples):
    # Class weighting shifts onset confidence; include high-confidence cutoffs.
    candidates = []
    for onset in (0.5, 0.7, 0.9, 0.95, 0.98, 0.995):
        for note in (0.05, 0.1, 0.2, 0.3, 0.5):
            params = {
                "threshold_onset": onset,
                "threshold_note": note,
                "min_duration_ms": 80,
                "tolerance_frames": 3,
            }
            score = score_predictions(examples, params)["f1"]
            candidates.append((score, params))
    candidates.sort(key=lambda item: item[0], reverse=True)
    best_score, best = candidates[0]
    for _, candidate in candidates[:5]:
        for duration in (40, 80, 120):
            for gap in (3, 11):
                params = {
                    **candidate,
                    "min_duration_ms": duration,
                    "tolerance_frames": gap,
                }
                score = score_predictions(examples, params)["f1"]
                if score > best_score:
                    best, best_score = params, score
    return best, best_score


def evaluate(
    checkpoint_path, device="auto", tune=False, max_windows=None, split="test"
):
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    if checkpoint["manifest_sha256"] != manifest_digest(config["manifest"]):
        raise ValueError("Evaluation manifest differs from training")
    device = choose_device(device)
    model = BasicPitchModel().to(device)
    model.load_state_dict(checkpoint["model"])
    run_dir = Path(checkpoint_path).resolve().parent
    decoder_file = run_dir / "decoder.json"
    thresholds = (
        json.loads(decoder_file.read_text())["thresholds"]
        if decoder_file.exists()
        else {"threshold_onset": 0.5, "threshold_note": 0.3}
    )
    if tune:
        validation = MaestroDataset(
            config["data_dir"], config["manifest"], "validation", config["cache_dir"]
        )
        val_predictions = collect_predictions(
            model, validation, device, config["batch_size"], max_windows
        )
        thresholds, score = tune_thresholds(val_predictions)
        decoder_file.write_text(
            json.dumps(
                {
                    "thresholds": thresholds,
                    "validation_note_f1": score,
                    "validation_windows": len(val_predictions),
                    "checkpoint": str(Path(checkpoint_path).resolve()),
                },
                indent=2,
            )
            + "\n"
        )
        print(f"Validation thresholds: {thresholds}, note F1={score:.4f}", flush=True)
    dataset = MaestroDataset(
        config["data_dir"], config["manifest"], split, config["cache_dir"]
    )
    examples = collect_predictions(
        model, dataset, device, config["batch_size"], max_windows
    )
    frame = {}
    for key in ("onset", "pitch", "note"):
        counts = np.zeros(3, dtype=np.int64)
        for example in examples:
            counts += frame_counts(
                torch.from_numpy(example["outputs"][key]),
                torch.from_numpy(example["labels"][key]),
                threshold=thresholds["threshold_onset"]
                if key == "onset"
                else thresholds["threshold_note"],
            )
        frame[key] = counts_to_metrics(*counts)
    result = {
        "split": split,
        "windows": len(examples),
        "thresholds": thresholds,
        "note_events_no_offset": score_predictions(examples, thresholds),
        "note_events_with_offset": score_predictions(examples, thresholds, True),
        "frame": frame,
        "scope": "Fixed held-out two-second windows; boundary notes are clipped.",
    }
    (run_dir / f"{split}_metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/demo/best.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.device, args.tune, args.max_windows, args.split)


if __name__ == "__main__":
    main()
