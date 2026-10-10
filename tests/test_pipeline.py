import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pretty_midi
import soundfile as sf
import torch
from audio_features import (
    FEATURES as C,
    compute_hcqt,
    labels_for_notes,
    load_window,
    pad_audio,
)
from MaestroDataset import MaestroDataset
from model import BasicPitchModel, compute_loss
from note_creator import NoteCreator, notes_to_midi
from performance_metrics import calculate_metrics, note_counts, tune_thresholds
from prepare_data import build_manifest
from project_config import load_checkpoint
from train import save_checkpoint


def note(pitch, start, end):
    return pretty_midi.Note(100, pitch, start, end)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_window_boundaries_and_subframe_notes(self):
        notes = [
            note(60, 0.5, 1.2),
            note(64, 1.0, 1.5),
            note(67, 2.0, 4.0),
            note(70, 3.0, 3.5),
            note(71, 4.0, 5.0),
            note(72, 1.101, 1.102),
        ]
        _, mask = pad_audio(np.zeros(C.n_samples))
        labels = labels_for_notes(notes, C.sample_rate, mask)
        self.assertEqual(labels["onset"][:, 60 - 21].sum(), 0)
        self.assertEqual(labels["note"][0, 60 - 21], 1)
        self.assertEqual(labels["onset"][0, 64 - 21], 1)
        self.assertEqual(labels["onset"][:, 70 - 21].sum(), 0)
        self.assertEqual(labels["onset"][:, 71 - 21].sum(), 0)
        self.assertEqual(labels["note"][-1, 67 - 21], 1)
        self.assertEqual(labels["onset"][:, 72 - 21].sum(), 1)
        self.assertTrue(np.all(labels["onset"] <= labels["note"]))

    def test_chord_and_full_piano_range(self):
        _, mask = pad_audio(np.zeros(C.n_samples))
        labels = labels_for_notes(
            [note(p, 0.2, 1.0) for p in (21, 60, 64, 67, 108)], 0, mask
        )
        self.assertEqual(labels["onset"].sum(), 5)
        self.assertEqual(labels["pitch"].shape, (173, 264))
        self.assertEqual(labels["note"].shape, (173, 88))
        for p in (21, 60, 64, 67, 108):
            np.testing.assert_array_equal(
                labels["pitch"][:, 3 * (p - 21)], labels["note"][:, p - 21]
            )

    def test_padding_is_masked(self):
        _, mask = pad_audio(np.zeros(500))
        labels = labels_for_notes([note(60, 0, 3.0)], 0, mask)
        self.assertEqual(mask.sum(), 2)
        self.assertFalse(labels["note"][~mask].any())
        model = BasicPitchModel()
        outputs = model(torch.zeros(1, 8, 173, 264), return_logits=True)
        targets = {k: torch.from_numpy(v)[None] for k, v in labels.items()}
        loss = compute_loss(outputs, targets)
        changed = {k: v.clone() for k, v in outputs.items()}
        for v in changed.values():
            v[:, 2:] = 100
        self.assertAlmostEqual(
            float(loss), float(compute_loss(changed, targets)), places=5
        )

    def test_hcqt_known_tone(self):
        time = np.arange(C.n_samples) / C.sample_rate
        for pitch in (21, 60, 69, 108):
            frequency = pretty_midi.note_number_to_hz(pitch)
            features = compute_hcqt(
                np.sin(2 * np.pi * frequency * time).astype(np.float32)
            )
            self.assertEqual(features.shape, (8, 173, 264))
            self.assertLessEqual(
                abs(int(features[1, 20:-20].mean(0).argmax()) - 3 * (pitch - 21)), 1
            )
            self.assertTrue(np.isfinite(features).all())
            self.assertTrue(((features >= 0) & (features <= 1)).all())

    def test_model_keeps_time_and_backward(self):
        model = BasicPitchModel()
        for frames in (17, 173):
            outputs = model(torch.rand(2, 8, frames, 264), return_logits=True)
            self.assertEqual(outputs["pitch"].shape, (2, frames, 264))
            self.assertEqual(outputs["onset"].shape, (2, frames, 88))
            targets = {k: torch.zeros_like(v) for k, v in outputs.items()}
            targets["mask"] = torch.ones(2, frames, dtype=torch.bool)
            compute_loss(outputs, targets).backward()
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in model.parameters()
            )
        )

    def test_metrics_cannot_broadcast_examples(self):
        pred = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        self.assertEqual(calculate_metrics(pred, pred)["f1"], 1)
        with self.assertRaises(ValueError):
            calculate_metrics(pred[:, None], pred)
        self.assertEqual(note_counts([(0.0, 1.0, 60)], [(0.0, 1.0, 60)]), (1, 1, 1))

    def test_decoder_gap_repeated_note_and_midi_roundtrip(self):
        onset = np.zeros((100, 88))
        activity = np.zeros_like(onset)
        activity[0:90, 39] = 0.8
        activity[40, 39] = 0
        onset[0, 39] = 0.9
        onset[50, 39] = 0.9
        original = activity.copy()
        notes, _ = NoteCreator().create_notes(onset, activity)
        self.assertEqual(len(notes), 2)
        self.assertTrue(all(n[2] == 60 for n in notes))
        self.assertAlmostEqual(notes[0][1], notes[1][0])
        np.testing.assert_array_equal(activity, original)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.mid"
            notes_to_midi(notes, path)
            restored = pretty_midi.PrettyMIDI(str(path)).instruments[0].notes
            self.assertEqual([n.pitch for n in restored], [60, 60])
            self.assertAlmostEqual(restored[0].start, notes[0][0], places=2)

    def test_single_dropout_does_not_split(self):
        onset = np.zeros((100, 88))
        activity = np.zeros_like(onset)
        onset[10, 39] = 0.9
        activity[10:90, 39] = 0.8
        activity[50, 39] = 0
        notes, _ = NoteCreator().create_notes(onset, activity)
        self.assertEqual(len(notes), 1)
        self.assertAlmostEqual(notes[0][1], 90 * C.frame_seconds)

    def test_seek_resample_and_short_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            rate = 44100
            time = np.arange(6 * rate) / rate
            audio = np.sin(2 * np.pi * np.where(time < 4, 440, 880) * time).astype(
                np.float32
            )
            path = Path(tmp) / "source.wav"
            sf.write(path, np.stack([audio, audio], axis=1), rate)
            window, mask = load_window(path, 4 * C.sample_rate)
            peak = np.fft.rfftfreq(len(window), 1 / C.sample_rate)[
                np.argmax(abs(np.fft.rfft(window)))
            ]
            self.assertAlmostEqual(peak, 880, delta=1)
            self.assertEqual(mask.sum(), 173)
            tail, tail_mask = load_window(path, 6 * C.sample_rate - 2205)
            self.assertEqual(tail_mask.sum(), 9)
            self.assertFalse(tail[2205:].any())

    def test_overlap_inference_preserves_absolute_time(self):
        from transcribe import predict_audio

        class ConstantModel(torch.nn.Module):
            def forward(self, x):
                return {
                    key: torch.full((len(x), 173, bins), 0.8)
                    for key, bins in [("onset", 88), ("note", 88), ("pitch", 264)]
                }

        audio = np.zeros(3 * C.n_samples + 123, dtype=np.float32)
        with patch(
            "transcribe.compute_hcqt", return_value=np.zeros((8, 173, 264), np.float32)
        ):
            result = predict_audio(ConstantModel(), audio, torch.device("cpu"))
        for key, bins in [("onset", 88), ("note", 88), ("pitch", 264)]:
            self.assertEqual(result[key].shape, (int(np.ceil(len(audio) / 256)), bins))
            np.testing.assert_allclose(result[key], 0.8, atol=1e-6)

    def test_calibration_handles_weighted_onset_confidence(self):
        onset = np.zeros((100, 88))
        activity = np.zeros_like(onset)
        onset[10, 39] = 0.998
        onset[40, 39] = 0.95
        activity[10:90, 39] = 0.8
        examples = [
            {
                "outputs": {"onset": onset, "note": activity},
                "reference": [(10 * C.frame_seconds, 90 * C.frame_seconds, 60)],
            }
        ]
        thresholds, score = tune_thresholds(examples)
        self.assertGreaterEqual(thresholds["threshold_onset"], 0.98)
        self.assertEqual(score, 1)

    def test_safe_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            model = BasicPitchModel().eval()
            optimizer = torch.optim.Adam(model.parameters())
            path = root / "checkpoint.pt"
            save_checkpoint(
                path, model, optimizer, 0, 0.1, {"manifest": str(manifest)}, [], 0
            )
            saved = load_checkpoint(path)
            restored = BasicPitchModel().eval()
            restored.load_state_dict(saved["model"])
            inputs = torch.rand(1, 8, 17, 264)
            with torch.inference_mode():
                original = model(inputs)
                loaded = restored(inputs)
            for key in original:
                torch.testing.assert_close(original[key], loaded[key])

    def test_manifest_reproducibility_split_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split in ("train", "validation", "test"):
                audio = root / f"{split}.wav"
                midi_path = root / f"{split}.midi"
                sf.write(audio, np.zeros(C.sample_rate * 4), C.sample_rate)
                midi = pretty_midi.PrettyMIDI()
                inst = pretty_midi.Instrument(0)
                inst.notes = [note(60, 0.2, 0.8)]
                midi.instruments.append(inst)
                midi.write(str(midi_path))
                rows.append(
                    {
                        "split": split,
                        "audio_filename": audio.name,
                        "midi_filename": midi_path.name,
                        "duration": "4",
                        "canonical_composer": "Test",
                        "canonical_title": split,
                    }
                )
            counts = {s: 1 for s in ("train", "validation", "test")}
            windows = {s: 2 for s in counts}
            manifest = build_manifest(rows, root, counts, windows)
            self.assertEqual(manifest, build_manifest(rows, root, counts, windows))
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            dataset = MaestroDataset(root, path, "train", root / "cache")
            first = dataset[0]
            with patch.object(
                dataset, "_read", side_effect=AssertionError("cache missed")
            ):
                second = dataset[0]
            torch.testing.assert_close(first[0], second[0], atol=0.0005, rtol=0.001)
            # A corrupt cache is safely regenerated.
            dataset.cache_path(0).write_bytes(b"broken")
            dataset[0]

    def test_expanded_training_preserves_frozen_holdouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = []
            for split, count in (("train", 4), ("validation", 2), ("test", 2)):
                for index in range(count):
                    name = f"{split}-{index}"
                    sf.write(
                        root / f"{name}.wav", np.zeros(4 * C.sample_rate), C.sample_rate
                    )
                    (root / f"{name}.midi").touch()
                    rows.append(
                        {
                            "split": split,
                            "audio_filename": f"{name}.wav",
                            "midi_filename": f"{name}.midi",
                            "duration": "4",
                            "canonical_composer": "Test",
                            "canonical_title": name,
                        }
                    )
            counts = {split: 1 for split in ("train", "validation", "test")}
            windows = {split: 2 for split in counts}
            original = build_manifest(rows, root, counts, windows)
            expanded = build_manifest(
                rows, root, {**counts, "train": 3}, windows, holdout_manifest=original
            )
            self.assertEqual(len(expanded["splits"]["train"]), 3)
            for split in ("validation", "test"):
                self.assertEqual(expanded["splits"][split], original["splits"][split])
            original_train = {
                track["audio_filename"] for track in original["splits"]["train"]
            }
            expanded_train = {
                track["audio_filename"] for track in expanded["splits"]["train"]
            }
            self.assertTrue(original_train <= expanded_train)
            # A training recording cannot be injected into the validation split.
            incorrect_split = json.loads(json.dumps(original))
            incorrect_split["splits"]["validation"] = original["splits"]["train"]
            with self.assertRaises(ValueError):
                build_manifest(
                    rows, root, counts, windows, holdout_manifest=incorrect_split
                )
            invalid_window = json.loads(json.dumps(original))
            invalid_window["splits"]["test"][0]["start_samples"][0] = 4 * C.sample_rate
            with self.assertRaises(ValueError):
                build_manifest(
                    rows, root, counts, windows, holdout_manifest=invalid_window
                )


if __name__ == "__main__":
    unittest.main()
