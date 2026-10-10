import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
import pretty_midi
import soundfile as sf
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from torch.utils.data import DataLoader

from audio_features import FEATURES, labels_for_notes, pad_audio
from model import BasicPitchModel, compute_loss, compute_loss_components
from observability import TrainingObserver, preview_indices
from project_config import read_config
from train import run_epoch


class ObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_logging_preserves_updates_and_head_losses(self):
        torch.manual_seed(42)
        samples = []
        for _ in range(3):
            inputs = torch.rand(8, 17, 264)
            labels = {
                key: (torch.rand(17, bins) > 0.97).float()
                for key, bins in (("onset", 88), ("pitch", 264), ("note", 88))
            }
            labels["mask"] = torch.tensor([True] * 15 + [False] * 2)
            samples.append((inputs, labels))
        original = BasicPitchModel()
        observed = copy.deepcopy(original)
        loader = DataLoader(samples, batch_size=2)
        device = torch.device("cpu")
        baseline = run_epoch(
            original, loader, device, torch.optim.Adam(original.parameters())
        )
        observer = Mock()
        measured = run_epoch(
            observed,
            loader,
            device,
            torch.optim.Adam(observed.parameters()),
            observer=observer,
            start_step=10,
        )
        self.assertEqual(baseline["loss"], measured["loss"])
        self.assertAlmostEqual(
            measured["loss"], sum(measured["loss_components"].values()), places=6
        )
        for key, value in original.state_dict().items():
            torch.testing.assert_close(
                value, observed.state_dict()[key], rtol=0, atol=0
            )
        self.assertEqual(
            [call.args[0] for call in observer.log_batch.call_args_list], [11, 12]
        )
        batch_inputs, targets = next(iter(loader))
        logits = observed(batch_inputs, return_logits=True)
        parts = compute_loss_components(logits, targets)
        torch.testing.assert_close(sum(parts.values()), compute_loss(logits, targets))
        altered = {key: value.clone() for key, value in logits.items()}
        for values in altered.values():
            values[:, 15:] = 100
        for key, loss in compute_loss_components(altered, targets).items():
            torch.testing.assert_close(loss, parts[key])

    def test_validation_preview_event_files_media_and_split_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sf.write(
                root / "clip.wav", np.zeros(FEATURES.n_samples), FEATURES.sample_rate
            )
            manifest = root / "manifest.json"
            manifest.write_text("{}")
            config_path = root / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "manifest": str(manifest),
                        "observability": {"validation_windows": 2},
                    }
                )
            )
            config = read_config(config_path)
            _, mask = pad_audio(np.zeros(FEATURES.n_samples))
            source_note = pretty_midi.Note(
                100, 60, 10 * FEATURES.frame_seconds, 60 * FEATURES.frame_seconds
            )
            arrays = labels_for_notes([source_note], 0, mask)
            labels = {key: torch.from_numpy(value) for key, value in arrays.items()}

            class ValidationDataset:
                split = "validation"
                data_dir = root
                windows = [
                    ({"audio_filename": "clip.wav", "midi_filename": "clip.midi"}, 0)
                ] * 2
                _notes = {str(root / "clip.midi"): [source_note]}

                def __len__(self):
                    return 2

                def __getitem__(self, index):
                    return torch.zeros(8, 173, 264), labels

            class PerfectModel(torch.nn.Module):
                def forward(self, inputs):
                    return {
                        key: labels[key][None] for key in ("onset", "pitch", "note")
                    }

            dataset, model = ValidationDataset(), PerfectModel()
            self.assertTrue(model.training)
            with TrainingObserver(root, config, torch.device("cpu")) as observer:
                metrics = observer.validation_preview(
                    model, dataset, torch.device("cpu"), 1, 12
                )
                self.assertTrue(model.training)
                self.assertEqual(metrics["note_events"]["f1"], 1.0)
                self.assertEqual(metrics["note_events_with_offset"]["f1"], 1.0)
                observer.log_epoch(
                    {
                        "epoch": 1,
                        "global_step": 12,
                        "train_loss": 0.2,
                        "validation_loss": 0.1,
                        "validation_preview": metrics,
                    }
                )
                dataset.split = "test"
                with self.assertRaises(ValueError):
                    observer.validation_preview(
                        model, dataset, torch.device("cpu"), 1, 12
                    )
            events = EventAccumulator(str(root / "tensorboard")).Reload()
            self.assertEqual(events.Scalars("loss/total/train")[0].step, 12)
            self.assertEqual(
                events.Scalars("validation_preview/note_events/f1")[0].value, 1.0
            )
            self.assertIn("preview/piano_roll", events.Tags()["images"])
            self.assertEqual(
                set(events.Tags()["audio"]),
                {"audio/reference", "audio/predicted_simple_tones"},
            )
            self.assertIn("preview/confidence/onset", events.Tags()["histograms"])
            self.assertTrue((root / "preview/step-12.mid").is_file())
        self.assertEqual(preview_indices(160, 16)[-1], 159)
        self.assertEqual(len(set(preview_indices(3, 16))), 3)


if __name__ == "__main__":
    unittest.main()
