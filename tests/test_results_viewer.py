import unittest

from build_demo import note_diagnostics


class ResultsViewerTests(unittest.TestCase):
    def test_duration_diagnostics_distinguish_endings_and_extra_notes(self):
        reference = [(0.0, 0.5, 60), (1.0, 1.5, 64), (2.0, 2.5, 67)]
        predicted = [(0.02, 1.2, 60), (1.01, 1.55, 64), (3.0, 3.2, 72)]
        result = note_diagnostics(predicted, reference)
        self.assertEqual(result["onset_matches"], 2)
        self.assertEqual(result["ending_matches_among_onset_matches"], 1)
        self.assertEqual(result["extra_notes"], 1)
        self.assertEqual(result["missed_notes"], 1)
        self.assertEqual(result["with_offset"]["matched"], 1)
        error = result["duration_examples"][0]
        self.assertEqual(error["note"], "C4")
        self.assertAlmostEqual(error["end_error_seconds"], 0.7)

    def test_empty_predictions_count_missed_notes_without_duration_claims(self):
        result = note_diagnostics([], [(0.0, 0.5, 60)])
        self.assertEqual(result["missed_notes"], 1)
        self.assertEqual(result["with_offset"]["f1"], 0)
        self.assertEqual(result["duration_examples"], [])


if __name__ == "__main__":
    unittest.main()
