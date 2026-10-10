"""Transcribe arbitrary audio or a held-out example into MIDI and an inspection page."""

import argparse
import json
from pathlib import Path
import librosa
import numpy as np
import pretty_midi
import torch
from audio_features import FEATURES as C, compute_hcqt, pad_audio
from inspection_report import create_report
from model import BasicPitchModel
from note_creator import NoteCreator
from performance_metrics import counts_to_metrics, note_counts
from project_config import choose_device, load_checkpoint


def predict_audio(model, audio, device, batch_size=4):
    if not len(audio):
        raise ValueError("Audio is empty")
    n_frames = int(np.ceil(len(audio) / C.hop_length))
    step = C.n_frames // 2
    last = max(0, n_frames - C.n_frames)
    starts = sorted(set(list(range(0, last + 1, step)) + [last]))
    sums = {
        k: np.zeros((n_frames, bins), dtype=np.float32)
        for k, bins in [
            ("onset", C.n_notes),
            ("note", C.n_notes),
            ("pitch", C.n_pitch_bins),
        ]
    }
    weights = np.zeros(n_frames, dtype=np.float32)
    taper = (np.hanning(C.n_frames) + 0.05).astype(np.float32)
    model.eval()
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            group = starts[begin : begin + batch_size]
            features = []
            for start in group:
                padded, _ = pad_audio(
                    audio[start * C.hop_length : start * C.hop_length + C.n_samples]
                )
                features.append(
                    compute_hcqt(padded).astype(np.float16).astype(np.float32)
                )
            outputs = {
                k: v.cpu().numpy()
                for k, v in model(
                    torch.from_numpy(np.stack(features)).to(device)
                ).items()
            }
            for b, start in enumerate(group):
                count = min(C.n_frames, n_frames - start)
                w = taper[:count]
                weights[start : start + count] += w
                for key in sums:
                    sums[key][start : start + count] += (
                        outputs[key][b, :count] * w[:, None]
                    )
    return {k: v / weights[:, None] for k, v in sums.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/demo/best.pt")
    parser.add_argument("--audio")
    parser.add_argument("--reference")
    parser.add_argument("--start", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument(
        "--example", choices=["train", "validation", "test"], default="test"
    )
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--output", default="runs/demo/inspection")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    device = choose_device(args.device)
    torch.set_num_threads(4)
    model = BasicPitchModel().to(device)
    model.load_state_dict(checkpoint["model"])
    if args.audio:
        audio_path = Path(args.audio)
        reference_path = Path(args.reference) if args.reference else None
        offset = args.start or 0
        duration = args.duration
    else:
        track = checkpoint["manifest"]["splits"][args.example][args.example_index]
        audio_path = Path(config["data_dir"]) / track["audio_filename"]
        reference_path = Path(config["data_dir"]) / track["midi_filename"]
        offset = (
            args.start
            if args.start is not None
            else track["start_samples"][0] / C.sample_rate
        )
        duration = args.duration if args.duration is not None else 8.0
    if offset < 0 or duration is not None and duration <= 0:
        raise ValueError("Invalid start/duration")
    audio, _ = librosa.load(
        audio_path, sr=C.sample_rate, offset=offset, duration=duration
    )
    outputs = predict_audio(model, audio, device, config["batch_size"])
    decoder_file = Path(args.checkpoint).resolve().parent / "decoder.json"
    thresholds = (
        json.loads(decoder_file.read_text())["thresholds"]
        if decoder_file.exists()
        else {}
    )
    predicted, _ = NoteCreator(**thresholds).create_notes(
        outputs["onset"], outputs["note"]
    )
    end = len(audio) / C.sample_rate
    predicted = [(s, min(e, end), p) for s, e, p in predicted if s < end]
    reference = None
    metrics = None
    if reference_path:
        midi = pretty_midi.PrettyMIDI(str(reference_path))
        reference = [
            (max(0, n.start - offset), min(end, n.end - offset), n.pitch)
            for inst in midi.instruments
            if not inst.is_drum
            for n in inst.notes
            if n.end > offset
            and n.start < offset + end
            and C.min_midi <= n.pitch < C.min_midi + C.n_notes
        ]
        metrics = counts_to_metrics(*note_counts(predicted, reference))
    output = Path(args.output)
    title = (
        audio_path.stem
        if args.audio
        else f"{args.example.capitalize()} example — {track['canonical_composer']}: {track['canonical_title']}"
    )
    path = create_report(output, audio, predicted, reference, metrics, title)
    (output / "events.json").write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "offset": offset,
                "duration": end,
                "predicted": predicted,
                "reference": reference,
                "metrics": metrics,
                "thresholds": thresholds,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Predicted {len(predicted)} notes; inspection: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
