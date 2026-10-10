"""Run masked-loss and training-contract regressions."""

from pathlib import Path
import unittest

if __name__ == "__main__":
    suite = unittest.defaultTestLoader.discover(str(Path(__file__).parent / "tests"))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
