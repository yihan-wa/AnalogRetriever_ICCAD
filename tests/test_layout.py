"""Checkout-relative paths and script entry points."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


class LayoutChecks(unittest.TestCase):
    def test_artifact_root_does_not_follow_working_directory(self):
        env = dict(os.environ, PYTHONPATH=str(ROOT / 'src'))
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, '-c',
                 'from analogretriever.paths import repository_root; print(repository_root())'],
                cwd=cwd, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(Path(result.stdout.strip()), ROOT)

    def test_validation_command_from_another_directory(self):
        with tempfile.TemporaryDirectory() as cwd:
            for entry in ['scripts/validate_release.py']:
                result = subprocess.run([sys.executable, str(ROOT / entry)], cwd=cwd,
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('Validated split', result.stdout)

    def test_no_root_python_or_shell_wrappers(self):
        self.assertEqual(list(ROOT.glob('*.py')), [])
        self.assertEqual(list(ROOT.glob('*.sh')), [])
