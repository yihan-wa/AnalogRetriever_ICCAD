"""Tiny local regressions for split, checker and refinement bookkeeping."""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_dataset import prepare_dataset
from scripts.validate_release import validate_data
from data_refinement import stage1_refine as stage1
from data_refinement import stage2_refine as stage2
from data_refinement.pipeline_io import load_prompts
from data_refinement.spice_checker import check_compilation, check_dc_simulation, dc_only_netlist, preprocess_netlist

try:
    from PIL import Image
except ImportError:
    Image = None
try:
    import clip
except ImportError:
    clip = None

NETLIST = "V1 1 0 1\nR1 1 0 1k\n.options reltol=1e-4\n.end"
RESPONSE = f"<NETLIST>{NETLIST}</NETLIST><DESCRIPTION>A resistor load.</DESCRIPTION>"
CHECK = {"compile_success": True, "dc_pass": True}


class SplitChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.output, self.splits = [self.root / part for part in ("data", "output", "splits")]
        self.splits.mkdir()
        (self.splits / "train_ids.txt").write_text("3\n1\n")
        (self.splits / "test_ids.txt").write_text("2\n")
        for folder in ("images", "spice", "captions"):
            (self.data / folder).mkdir(parents=True)
        for index in (1, 2, 3):
            (self.data / "images" / f"img{index}.jpg").write_bytes(b"image placeholder")
            (self.data / "spice" / f"spice{index}.txt").write_text(NETLIST)
            (self.data / "captions" / f"cap{index}.txt").write_text(f"Circuit {index}")

    def test_preserves_committed_membership_and_order(self):
        prepare_dataset(self.data, self.output, self.splits)
        with (self.output / "train_data.csv").open() as handle:
            self.assertEqual([int(row["id"]) for row in csv.DictReader(handle)], [3, 1])
        with (self.output / "test_data.csv").open() as handle:
            self.assertEqual([int(row["id"]) for row in csv.DictReader(handle)], [2])

    def test_missing_modality_does_not_reassign_or_overwrite_splits(self):
        prepare_dataset(self.data, self.output, self.splits)
        before = (self.output / "test_data.csv").read_bytes()
        (self.data / "captions/cap2.txt").unlink()
        with self.assertRaises(FileNotFoundError):
            prepare_dataset(self.data, self.output, self.splits)
        self.assertEqual((self.output / "test_data.csv").read_bytes(), before)

    @unittest.skipIf(Image is None, "Pillow is required for image validation")
    def test_nonempty_corrupt_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid image"):
            validate_data(self.data, [1, 2, 3])

    @unittest.skipIf(Image is None, "Pillow is required for image validation")
    def test_png_jpeg_images_work_and_duplicate_ids_are_rejected(self):
        for index, suffix in ((1, '.png'), (2, '.jpeg'), (3, '.jpg')):
            (self.data / 'images' / f'img{index}.jpg').unlink()
            Image.new('RGB', (2, 2)).save(self.data / 'images' / f'img{index}{suffix}')
        validate_data(self.data, [1, 2, 3])
        prepare_dataset(self.data, self.output, self.splits)
        Image.new('RGB', (2, 2)).save(self.data / 'images/img1.jpg')
        with self.assertRaisesRegex(ValueError, 'exactly one image'):
            validate_data(self.data, [1, 2, 3])


class CheckerChecks(unittest.TestCase):
    def test_device_first_title_and_options_are_preserved(self):
        processed = dc_only_netlist(NETLIST)
        self.assertTrue(processed.startswith("* SPICE Netlist\nV1 1 0 1\n"))
        self.assertIn(".options reltol=1e-4", processed)
        self.assertEqual(processed.splitlines().count(".op"), 1)

    def test_prose_title_is_preserved(self):
        for title in ("Voltage divider test", "Voltage divider circuit example"):
            with self.subTest(title=title):
                self.assertTrue(preprocess_netlist(title + "\n" + NETLIST).startswith(title + "\nV1"))

    def test_empty_input_is_rejected_without_launching_ngspice(self):
        with patch("data_refinement.spice_checker.subprocess.run") as run:
            for text in ("", "* empty\n.end", "spice\n.options reltol=1e-4\n.end", "spice\n.end\nR1 1 0 1k"):
                self.assertFalse(check_compilation(text)["compile_success"])
                self.assertFalse(check_dc_simulation(text)["dc_pass"])
            run.assert_not_called()

    def test_simulator_crash_cannot_count_as_compilation_success(self):
        crashed = subprocess.CompletedProcess([], -6, "", "Assertion failed")
        with patch("data_refinement.spice_checker.subprocess.run", return_value=crashed):
            self.assertFalse(check_compilation(NETLIST)["compile_success"])

    def test_dc_check_removes_other_analyses_and_control_blocks(self):
        text = "spice\nV1 1 0 1\nR1 1 0 1k\n.tran bad bad\n.print tran v(1)\n.control\nquit 1\n.endc\n.end"
        processed = dc_only_netlist(text)
        for command in (".tran", ".print", "quit", ".control"):
            self.assertNotIn(command, processed)
        self.assertIn(".op\n.end", processed)

    def test_recovered_singular_warning_does_not_override_successful_op(self):
        success = subprocess.CompletedProcess([], 0, "No. of Data Rows : 1\n", "Warning: singular matrix: check node 2\n")
        with patch("data_refinement.spice_checker.subprocess.run", return_value=success):
            self.assertTrue(check_dc_simulation(NETLIST)["dc_pass"])
        failed = subprocess.CompletedProcess([], 1, "", "Error: The operating point could not be simulated successfully.")
        with patch("data_refinement.spice_checker.subprocess.run", return_value=failed):
            self.assertFalse(check_dc_simulation(NETLIST)["dc_pass"])

    @unittest.skipUnless(shutil.which("ngspice"), "Local ngspice is optional")
    def test_ngspice_keeps_the_first_voltage_source(self):
        result = check_dc_simulation(NETLIST)
        self.assertTrue(result["dc_pass"], result["dc_errors"])
        self.assertRegex(result["dc_output"], r"V\(1\)\s+1\.0+e\+00")

    @unittest.skipUnless(shutil.which("ngspice"), "Local ngspice is optional")
    def test_malformed_first_device_is_not_swallowed_as_a_title(self):
        malformed = "R1 1 0\nV1 1 0 1\n.end"
        self.assertIn("\nR1 1 0\n", preprocess_netlist(malformed))
        self.assertFalse(check_compilation(malformed)["compile_success"])


class RefinementChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data, self.first, self.second = [self.root / part for part in ("data", "stage1", "stage2")]
        (self.data / "images").mkdir(parents=True)
        (self.data / "images/img1.jpg").write_bytes(b"mock image")
        self.mapping = self.data / "data_mapping.json"
        self.mapping.write_text("[]")

    def run_first(self):
        return stage1.run_stage1(self.data, self.mapping, self.first, workers=1, delay=0)

    def test_api_failure_retries_and_only_complete_triplets_enter_mapping(self):
        with patch.object(stage1, "call_llm", side_effect=[None, RESPONSE]) as api, patch.object(stage1, "check_netlist", return_value=CHECK):
            self.assertEqual(self.run_first()["failed"], 1)
            self.assertEqual(json.loads((self.first / "data_mapping.json").read_text()), [])
            self.assertEqual(self.run_first()["failed"], 0)
            self.assertEqual(api.call_count, 2)
        mapping = json.loads((self.first / "data_mapping.json").read_text())
        self.assertEqual(len(mapping), 1)
        for name in mapping[0].values():
            self.assertTrue((self.first / name).is_file())
        progress = json.loads((self.first / "stage1_progress.json").read_text())
        self.assertEqual(len(progress["completed"]), 1)
        self.assertEqual(progress["failed"], {})

    def test_resume_rechecks_files_instead_of_trusting_completed_key(self):
        with patch.object(stage1, "call_llm", return_value=RESPONSE) as api, patch.object(stage1, "check_netlist", return_value=CHECK):
            self.run_first()
            self.run_first()
            self.assertEqual(api.call_count, 1)
            (self.first / "captions/cap1.txt").unlink()
            self.run_first()
            self.assertEqual(api.call_count, 2)

    def test_current_progress_missing_image_is_copied_without_another_api_call(self):
        with patch.object(stage1, "call_llm", return_value=RESPONSE) as api, patch.object(stage1, "check_netlist", return_value=CHECK):
            self.run_first()
            (self.first / "images/img1.jpg").unlink()
            self.run_first()
            self.assertEqual(api.call_count, 1)
        self.assertTrue((self.first / "images/img1.jpg").is_file())

    def test_missing_description_is_retryable_failure(self):
        with patch.object(stage1, "call_llm", return_value=f"<NETLIST>{NETLIST}</NETLIST>"):
            self.assertEqual(self.run_first()["failed"], 1)
        self.assertEqual(json.loads((self.first / "data_mapping.json").read_text()), [])

    def test_stage2_exports_images_and_verified_intersection(self):
        with patch.object(stage1, "call_llm", return_value=RESPONSE), patch.object(stage1, "check_netlist", return_value=CHECK):
            self.run_first()
        repaired = {**CHECK, "netlist": NETLIST, "iterations": 0}
        with patch.object(stage2, "repair_netlist", return_value=repaired), patch.object(stage2, "refine_description", return_value="Refined circuit"):
            result = stage2.run_stage2(self.data, self.first, self.second, workers=1, delay=0)
        self.assertEqual(result["both_pass"], 1)
        mapping = json.loads((self.second / "data_mapping.json").read_text())
        for name in mapping[0].values():
            self.assertTrue((self.second / name).is_file())

    def test_stage2_repair_api_failure_retries_on_resume(self):
        with patch.object(stage1, 'call_llm', return_value=RESPONSE), patch.object(stage1, 'check_netlist', return_value=CHECK):
            self.run_first()
        bad = {'compile_success': False, 'dc_pass': False, 'compile_errors': 'Invalid card'}
        with patch.object(stage2, 'call_llm', side_effect=[None, RESPONSE]) as api, \
             patch.object(stage2, 'check_netlist', side_effect=[bad, bad, CHECK]), \
             patch.object(stage2, 'refine_description', return_value='Valid caption'), \
             patch.object(stage2.time, 'sleep'):
            first = stage2.run_stage2(self.data, self.first, self.second, max_iters=1, workers=1, delay=0)
            self.assertEqual(first['failed'], 1)
            self.assertEqual(json.loads((self.second / 'data_mapping.json').read_text()), [])
            second = stage2.run_stage2(self.data, self.first, self.second, max_iters=1, workers=1, delay=0)
            self.assertEqual(second['failed'], 0)
            self.assertEqual(api.call_count, 2)

    def test_valid_repair_attempt_can_be_reported_as_quality_failure(self):
        bad = {'compile_success': False, 'dc_pass': False, 'compile_errors': 'Invalid card'}
        with patch.object(stage2, 'call_llm', return_value=RESPONSE), \
             patch.object(stage2, 'check_netlist', return_value=bad), patch.object(stage2.time, 'sleep'):
            result = stage2.repair_netlist(NETLIST, 'unused.jpg', max_iters=1)
        self.assertFalse(result['compile_success'])
        self.assertEqual(result['netlist'], NETLIST)

    @unittest.skipIf(clip is None, 'CLIP tokenizer dependencies are required')
    def test_long_description_is_regenerated_without_silent_truncation(self):
        long_caption = 'resistor ' * 80
        short_caption = 'A common-source amplifier.'
        with patch.object(stage2, 'call_llm', side_effect=[f'<DESCRIPTION>{long_caption}</DESCRIPTION>',
                                                        f'<DESCRIPTION>{short_caption}</DESCRIPTION>']) as api:
            result = stage2.refine_description(NETLIST, 'Original caption')
        self.assertEqual(result, short_caption)
        self.assertEqual(api.call_count, 2)
        clip.tokenize([result], context_length=77, truncate=False)

    @unittest.skipIf(clip is None, 'CLIP tokenizer dependencies are required')
    def test_long_description_retry_budget_is_finite(self):
        response = '<DESCRIPTION>' + 'resistor ' * 80 + '</DESCRIPTION>'
        with patch.object(stage2, 'call_llm', return_value=response) as api:
            with self.assertRaisesRegex(ValueError, '77 CLIP tokens'):
                stage2.refine_description(NETLIST, 'Original caption', max_attempts=2)
        self.assertEqual(api.call_count, 2)

    @unittest.skipIf(clip is None, 'CLIP tokenizer dependencies are required')
    def test_stage2_description_api_failure_retries_on_resume(self):
        with patch.object(stage1, 'call_llm', return_value=RESPONSE), patch.object(stage1, 'check_netlist', return_value=CHECK):
            self.run_first()
        repaired = {**CHECK, 'netlist': NETLIST, 'iterations': 0}
        with patch.object(stage2, 'repair_netlist', return_value=repaired), \
             patch.object(stage2, 'check_netlist', return_value={'compile_success': False, 'dc_pass': False}), \
             patch.object(stage2, 'call_llm', side_effect=[None, None, None, '<DESCRIPTION>A resistor load.</DESCRIPTION>']) as api:
            first = stage2.run_stage2(self.data, self.first, self.second, workers=1, delay=0)
            self.assertEqual(first['failed'], 1)
            self.assertEqual(json.loads((self.second / 'data_mapping.json').read_text()), [])
            second = stage2.run_stage2(self.data, self.first, self.second, workers=1, delay=0)
            self.assertEqual(second['failed'], 0)
            self.assertEqual(api.call_count, 4)

    def test_executable_prompts_are_the_documented_english_blocks(self):
        self.assertEqual(load_prompts("stage1_prompt.md"), (stage1.SYSTEM_PROMPT, stage1.USER_PROMPT))
        self.assertEqual(load_prompts("stage2_netlist_repair_prompt.md"), (stage2.REPAIR_SYSTEM_PROMPT, stage2.REPAIR_USER_TEMPLATE))
        self.assertIn(".op analysis", stage2.REPAIR_USER_TEMPLATE)

    def test_stage2_description_prompt_follows_paper_figure5(self):
        for requirement in ("circuit type", "key components and their roles", "operating principle",
                            "77 tokens"):
            self.assertIn(requirement, stage2.DESC_USER_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
