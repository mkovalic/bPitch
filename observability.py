"""Local TensorBoard metrics and fixed validation previews; no test-set access."""

import argparse
import json
import math
from pathlib import Path
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from audio_features import FEATURES, load_window
from note_creator import NoteCreator, notes_to_midi
from performance_metrics import counts_to_metrics, note_counts, reference_notes
from project_config import load_checkpoint, manifest_digest


def preview_indices(size, count):
    """Evenly cover the fixed validation manifest, including its last recording."""
    if size <= 0 or count <= 0:
        raise ValueError("Validation preview requires positive window counts")
    return np.linspace(0, size - 1, min(size, count), dtype=int).tolist()


def write_history_row(writer, row, fallback_step):
    step = row.get("global_step", fallback_step)
    writer.add_scalar("progress/epoch", row["epoch"], step)
    for split, prefix in (("train", "train"), ("validation", "validation")):
        writer.add_scalar(f"loss/total/{split}", row[f"{prefix}_loss"], step)
        for key, value in row.get(f"{prefix}_loss_components", {}).items():
            writer.add_scalar(f"loss/{key}/{split}", value, step)
        for key in ("seconds", "loader_wait_seconds", "examples_per_second"):
            if f"{prefix}_{key}" in row:
                writer.add_scalar(
                    f"performance/{key}/{split}", row[f"{prefix}_{key}"], step
                )
    for key in ("learning_rate", "gradient_norm", "best_validation_loss", "bad_epochs"):
        if key in row:
            writer.add_scalar(f"training/{key}", row[key], step)
    for group, values in row.get("validation_preview", {}).items():
        if isinstance(values, dict):
            for key in ("precision", "recall", "f1"):
                writer.add_scalar(
                    f"validation_preview/{group}/{key}", values[key], step
                )
        elif isinstance(values, (int, float)):
            writer.add_scalar(f"validation_preview/{group}", values, step)


class TrainingObserver:
    def __init__(self, run_dir, config, device, resume_step=None):
        self.run_dir = Path(run_dir)
        self.settings = config["observability"]
        for key in ("log_every_batches", "validation_windows", "media_every_epochs"):
            if self.settings[key] <= 0:
                raise ValueError(f"observability.{key} must be positive")
        self.writer = SummaryWriter(
            str(self.run_dir / "tensorboard"),
            purge_step=resume_step + 1 if resume_step is not None else None,
            flush_secs=10,
        )
        self.writer.add_text(
            "run/config",
            "```json\n" + json.dumps(config, indent=2) + "\n```",
            resume_step or 0,
        )
        self.writer.add_text(
            "run/data_contract",
            json.dumps(
                {
                    "device": str(device),
                    "features": FEATURES.to_dict(),
                    "manifest_sha256": manifest_digest(config["manifest"]),
                    "step_axis": "Optimizer updates, including across resumes",
                    "validation_preview": "Fixed, evenly spaced validation windows; 50 ms onset / 50 cent pitch tolerance. Boundary notes are clipped. Decoder thresholds remain fixed during training; checkpoint selection still uses full validation loss.",
                },
                indent=2,
            ),
            resume_step or 0,
        )

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.writer.close()

    def log_batch(
        self,
        step,
        batch,
        values,
        gradient_norm,
        learning_rate,
        wait_seconds,
        processing_seconds,
    ):
        if batch != 1 and batch % self.settings["log_every_batches"]:
            return
        for key, value in values.items():
            self.writer.add_scalar(f"batch_loss/{key}", value, step)
        self.writer.add_scalar(
            "batch/gradient_norm_before_clipping", gradient_norm, step
        )
        self.writer.add_scalar("batch/learning_rate", learning_rate, step)
        self.writer.add_scalar("batch/loader_wait_seconds", wait_seconds, step)
        self.writer.add_scalar("batch/processing_seconds", processing_seconds, step)

    def validation_preview(self, model, dataset, device, epoch, step):
        if dataset.split != "validation":
            raise ValueError("Training previews must use the validation split")
        indices = preview_indices(len(dataset), self.settings["validation_windows"])
        decoder = NoteCreator(**self.settings["decoder"])
        totals = {
            "note_events": np.zeros(3, np.int64),
            "note_events_with_offset": np.zeros(3, np.int64),
        }
        was_training = model.training
        model.eval()
        started = time.perf_counter()
        preview = None
        try:
            with torch.inference_mode():
                for index in indices:
                    inputs, labels = dataset[index]
                    valid = labels["mask"].numpy()
                    outputs = {
                        key: value[0].cpu().numpy()[valid]
                        for key, value in model(inputs[None].to(device)).items()
                    }
                    predicted, _ = decoder.create_notes(
                        outputs["onset"], outputs["note"]
                    )
                    reference = reference_notes(dataset, index)
                    for key, offsets in (
                        ("note_events", False),
                        ("note_events_with_offset", True),
                    ):
                        totals[key] += note_counts(predicted, reference, offsets)
                    if preview is None:
                        preview = (index, outputs, predicted, reference)
        finally:
            model.train(was_training)
        metrics = {key: counts_to_metrics(*counts) for key, counts in totals.items()}
        metrics.update(windows=len(indices), seconds=time.perf_counter() - started)
        self.writer.add_text(
            "validation_preview/window_indices", json.dumps(indices), step
        )
        if epoch == 1 or epoch % self.settings["media_every_epochs"] == 0:
            self.log_media(dataset, preview, step)
        return metrics

    def log_media(self, dataset, preview, step):
        index, outputs, predicted, reference = preview
        track, start = dataset.windows[index]
        audio, _ = load_window(dataset.data_dir / track["audio_filename"], start)
        self.writer.add_audio(
            "audio/reference", audio[None], step, FEATURES.sample_rate
        )
        media_dir = self.run_dir / "preview"
        media_dir.mkdir(exist_ok=True)
        midi = notes_to_midi(
            predicted, media_dir / f"step-{step}.mid", FEATURES.window_seconds
        )
        synthesized = midi.synthesize(fs=FEATURES.sample_rate)
        synthesized = np.pad(synthesized, (0, max(0, len(audio) - len(synthesized))))[
            : len(audio)
        ]
        peak = np.max(np.abs(synthesized)) if len(synthesized) else 0
        if peak:
            synthesized = synthesized * (0.8 / peak)
        self.writer.add_audio(
            "audio/predicted_simple_tones",
            synthesized[None],
            step,
            FEATURES.sample_rate,
        )
        fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, sharey=True)
        all_notes = predicted + reference
        low = max(21, min((n[2] for n in all_notes), default=48) - 4)
        high = min(108, max((n[2] for n in all_notes), default=84) + 4)
        for ax, label, notes, color in (
            (axes[0], "Reference MIDI", reference, "#60738c"),
            (axes[1], "Predicted MIDI", predicted, "#137c66"),
        ):
            for first, last, pitch in notes:
                ax.add_patch(
                    Rectangle((first, pitch - 0.4), last - first, 0.8, facecolor=color)
                )
            ax.set_ylim(low - 0.5, high + 0.5)
            ax.set_xlim(0, FEATURES.window_seconds)
            ax.set_ylabel(label + "\nMIDI pitch")
            ax.grid(alpha=0.15)
        axes[-1].set_xlabel("Seconds")
        fig.tight_layout()
        self.writer.add_figure("preview/piano_roll", fig, step)
        for key in ("onset", "note", "pitch"):
            self.writer.add_histogram(f"preview/confidence/{key}", outputs[key], step)
        self.writer.add_text(
            "preview/source",
            json.dumps(
                {
                    "split": "validation",
                    "window_index": index,
                    "audio": track["audio_filename"],
                    "start_sample": start,
                    "decoder": self.settings["decoder"],
                },
                indent=2,
            ),
            step,
        )

    def log_epoch(self, row):
        write_history_row(self.writer, row, row["global_step"])
        self.writer.flush()


def import_run(run_dir):
    """Import only measurements that older runs actually saved."""
    run_dir = Path(run_dir).resolve()
    log_dir = run_dir / "tensorboard"
    if any(log_dir.glob("events.out.tfevents.*")):
        raise FileExistsError(f"TensorBoard events already exist: {log_dir}")
    config = json.loads((run_dir / "config.json").read_text())
    history = json.loads((run_dir / "history.json").read_text())
    manifest = json.loads(Path(config["manifest"]).read_text())
    windows = sum(len(track["start_samples"]) for track in manifest["splits"]["train"])
    steps_per_epoch = math.ceil(windows / config["batch_size"])
    with SummaryWriter(str(log_dir)) as writer:
        writer.add_text(
            "run/source",
            "Imported saved history and evaluation artifacts. Legacy step counts are inferred assuming full training epochs; timing, gradients, and head losses were not recorded.",
            0,
        )
        writer.add_text(
            "run/config", "```json\n" + json.dumps(config, indent=2) + "\n```", 0
        )
        for row in history:
            write_history_row(writer, row, row["epoch"] * steps_per_epoch)
        checkpoint = load_checkpoint(run_dir / "best.pt")
        step = checkpoint.get(
            "global_step", (checkpoint["epoch"] + 1) * steps_per_epoch
        )
        test_path = run_dir / "test_metrics.json"
        if test_path.exists():
            result = json.loads(test_path.read_text())
            for group in ("note_events_no_offset", "note_events_with_offset"):
                for key in ("precision", "recall", "f1"):
                    writer.add_scalar(
                        f"evaluation/test/{group}/{key}", result[group][key], step
                    )
            writer.add_text("evaluation/test/scope", result["scope"], step)
    return log_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--import-run", required=True, help="Import an older run's saved history once"
    )
    args = parser.parse_args()
    print(f"TensorBoard history: {import_run(args.import_run)}")


if __name__ == "__main__":
    main()
