"""Small mocked regressions for the original-repair Stage 1 protocol."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_refinement import stage1_refine as stage1
from data_refinement.pipeline_io import content_fingerprint, is_current, load_progress


ORIGINAL = "spice\nV1 1 0 1\nR1 1 0\n.end"
REPAIRED = "spice\nV1 1 0 1\nR1 1 0 1k\n.op\n.end"
CAPTION = "  Original functional description.\n"
RESPONSE = f"<NETLIST>{REPAIRED}</NETLIST><DESCRIPTION>Generated description.</DESCRIPTION>"
PASS = {"compile_success": True, "compile_errors": "", "dc_pass": True, "dc_errors": ""}
FAIL = {"compile_success": False, "compile_errors": "R1 missing resistance",
        "dc_pass": False, "dc_errors": "operating point failed"}


class Stage1ProtocolChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw, self.output = self.root / "raw", self.root / "output"
        for folder in ("images", "spice", "captions"):
            (self.raw / folder).mkdir(parents=True)
        for index in (1, 2):
            (self.raw / f"images/img{index}.png").write_bytes(f"image {index}".encode())
        (self.raw / "spice/original1.txt").write_text(ORIGINAL)
        (self.raw / "captions/original1.txt").write_text(CAPTION)
        self.mapping = self.raw / "data_mapping.json"
        self.entries = [{"image": "images/img1.png", "spice": "spice/original1.txt",
                         "caption": "captions/original1.txt"}]
        self.write_mapping()
        self.checker = patch.object(stage1, "check_netlist", return_value=PASS)
        self.checker.start()
        self.addCleanup(self.checker.stop)
        # Accidentally unmocked calls must fail before a credential lookup or network request.
        no_live_api = patch.object(stage1.llm_client, "get_api_key",
                                   side_effect=AssertionError("Live API calls forbidden in tests"))
        no_live_api.start()
        self.addCleanup(no_live_api.stop)

    def write_mapping(self):
        self.mapping.write_text(json.dumps(self.entries))

    def run_first(self, **kwargs):
        return stage1.run_stage1(self.raw, self.mapping, self.output,
                                 workers=1, delay=0, **kwargs)

    def progress(self):
        return json.loads((self.output / "stage1_progress.json").read_text())

    def test_original_repair_uses_original_errors_and_keeps_caption(self):
        original_failure = {**FAIL, "raw_output": "Detailed compilation diagnostics",
                            "dc_output": "Detailed operating point diagnostics"}
        def check(netlist):
            return original_failure if netlist == ORIGINAL else PASS

        with patch.object(stage1, "check_netlist", side_effect=check), \
             patch.object(stage1, "call_llm", return_value=RESPONSE) as api:
            result = self.run_first()
        self.assertEqual(api.call_count, 2)
        repair, generation = [call.kwargs for call in api.call_args_list]
        self.assertIn(ORIGINAL, repair["user_prompt"])
        self.assertIn(FAIL["compile_errors"], repair["user_prompt"])
        self.assertIn(FAIL["dc_errors"], repair["user_prompt"])
        self.assertIn(original_failure["raw_output"], repair["user_prompt"])
        self.assertIn(original_failure["dc_output"], repair["user_prompt"])
        self.assertNotIn(ORIGINAL, generation["user_prompt"])
        self.assertEqual(Path(repair["image_path"]), (self.raw / "images/img1.png").resolve())
        self.assertEqual((self.output / "captions/cap1.txt").read_text(), CAPTION)
        self.assertEqual((self.raw / "spice/original1.txt").read_text(), ORIGINAL)
        self.assertEqual((self.output / "captions/cap2.txt").read_text(), "Generated description.")
        self.assertEqual(result["repair_existing"], 1)
        self.assertEqual(result["generate_missing"], 1)
        progress = self.progress()
        self.assertEqual(progress["protocol_version"], stage1.STAGE1_PROTOCOL_VERSION)
        original = progress["completed"]["images/img1.png"]
        self.assertEqual(original["source_mode"], "repair_existing")
        self.assertEqual(original["caption_source"], "original")
        self.assertEqual(original["initial_check"], original_failure)
        self.assertEqual(original["input_netlist"], ORIGINAL)
        self.assertEqual(original["input_caption"], CAPTION)
        self.assertEqual(original["source_inputs"]["original_caption"],
                         str((self.raw / "captions/original1.txt").resolve()))
        self.assertEqual(progress["completed"]["images/img2.png"]["source_mode"], "generate_missing")
        for entry in json.loads((self.output / "data_mapping.json").read_text()):
            self.assertEqual(set(entry), {"image", "spice", "caption"})

    def test_passing_original_still_gets_one_initial_repair(self):
        with patch.object(stage1, "call_llm", return_value=f"<NETLIST>{REPAIRED}</NETLIST>") as api:
            result = self.run_first(end=1)
        self.assertEqual(api.call_count, 1)
        self.assertEqual(result["total"], 1)
        self.assertEqual(result["failed"], 0)
        self.assertEqual(result["protocol_version"], stage1.STAGE1_PROTOCOL_VERSION)
        self.assertEqual((self.output / "captions/cap1.txt").read_text(), CAPTION)

    def test_mapped_missing_files_fail_before_any_api_call(self):
        for relative in ("images/img1.png", "spice/original1.txt", "captions/original1.txt"):
            with self.subTest(relative=relative):
                path = self.raw / relative
                content = path.read_bytes()
                path.unlink()
                try:
                    with patch.object(stage1, "call_llm") as api, \
                         self.assertRaisesRegex(ValueError, "preflight failed"):
                        self.run_first()
                    api.assert_not_called()
                    self.assertFalse(self.output.exists())
                finally:
                    path.write_bytes(content)

    def test_blank_original_text_is_not_treated_as_unmapped(self):
        for relative in ("spice/original1.txt", "captions/original1.txt"):
            path = self.raw / relative
            original = path.read_text()
            path.write_text(" \n\t")
            with self.assertRaisesRegex(ValueError, "blank text"):
                stage1.build_image_list(self.raw, self.mapping)
            path.write_text(original)

    def test_duplicate_image_ids_and_mapping_are_rejected(self):
        extra = self.raw / "images/img01.jpg"
        extra.write_bytes(b"another image")
        with self.assertRaisesRegex(ValueError, "Duplicate image ID"):
            stage1.build_image_list(self.raw, self.mapping)
        extra.unlink()
        self.entries.append(dict(self.entries[0]))
        self.write_mapping()
        with self.assertRaisesRegex(ValueError, "Duplicate image mapping"):
            stage1.build_image_list(self.raw, self.mapping)

    def test_missing_mapping_is_never_assumed_empty(self):
        self.mapping.unlink()
        with self.assertRaisesRegex(ValueError, "Missing or empty original mapping"):
            stage1.build_image_list(self.raw, self.mapping)

    def test_legacy_progress_is_refused(self):
        self.output.mkdir()
        (self.output / "stage1_progress.json").write_text('{"completed": {}}')
        with patch.object(stage1, "call_llm") as api, \
             self.assertRaisesRegex(ValueError, "legacy progress.*new output directory"):
            self.run_first()
        api.assert_not_called()

    def test_unchanged_resume_and_lost_image_do_not_repeat_paid_requests(self):
        with patch.object(stage1, "call_llm", return_value=RESPONSE) as api:
            self.run_first()
            self.run_first()
            self.assertEqual(api.call_count, 2)
            (self.output / "images/img1.png").unlink()
            self.run_first()
            self.assertEqual(api.call_count, 2)
        record = self.progress()["completed"]["images/img1.png"]
        self.assertTrue(is_current(self.output, record, record["input_fingerprint"]))

    def test_original_and_output_content_changes_invalidate_resume(self):
        with patch.object(stage1, "call_llm", return_value=RESPONSE) as api:
            self.run_first()
            for count, (relative, value) in enumerate((
                    ("spice/original1.txt", ORIGINAL + "\n* changed"),
                    ("captions/original1.txt", "Changed original caption."),
                    ("images/img1.png", "changed image bytes")), 3):
                (self.raw / relative).write_text(value)
                self.run_first()
                self.assertEqual(api.call_count, count)
            (self.output / "captions/cap1.txt").write_text("Unexpected output edit")
            self.run_first()
            self.assertEqual(api.call_count, 6)

    def test_other_mapping_entry_change_only_invalidates_its_sample(self):
        with patch.object(stage1, "call_llm", return_value=RESPONSE) as api:
            self.run_first()
            (self.raw / "spice/original2.txt").write_text(REPAIRED)
            (self.raw / "captions/original2.txt").write_text("Second original caption")
            self.entries.append({"image": "images/img2.png", "spice": "spice/original2.txt",
                                 "caption": "captions/original2.txt"})
            self.write_mapping()
            self.run_first()
        self.assertEqual(api.call_count, 3)
        self.assertTrue(api.call_args.kwargs["image_path"].endswith("img2.png"))

    def test_model_and_executable_prompt_changes_invalidate_resume(self):
        with patch.object(stage1, "call_llm", return_value=RESPONSE) as api:
            self.run_first()
            with patch.object(stage1.llm_client, "MODEL_NAME", "test/new-model"):
                self.run_first()
            self.assertEqual(api.call_count, 4)
            original_loader = stage1.load_prompts
            with patch.object(stage1, "load_prompts", side_effect=lambda name:
                              tuple(text + "\nChanged prompt." for text in original_loader(name))):
                self.run_first()
                self.assertIn("Changed prompt.", api.call_args.kwargs["user_prompt"])
            self.assertEqual(api.call_count, 6)

    def test_repair_api_failure_remains_retryable_and_simulation_failure_is_exported(self):
        with patch.object(stage1, "call_llm", side_effect=[None, RESPONSE]) as api, \
             patch.object(stage1, "check_netlist", return_value=FAIL):
            failed = self.run_first(end=1)
            self.assertEqual(failed["failed"], 1)
            self.assertEqual(json.loads((self.output / "data_mapping.json").read_text()), [])
            recovered = self.run_first(end=1)
        self.assertEqual(api.call_count, 2)
        self.assertEqual(recovered["failed"], 0)
        self.assertEqual(recovered["total"], 1)
        self.assertEqual(recovered["both_pass"], 0)

    def test_validation_cli_needs_no_output_directory_or_api(self):
        argv = ["stage1", "--data-dir", str(self.raw), "--mapping", str(self.mapping),
                "--validate-only", "--expected-original", "1", "--expected-generated", "1"]
        output = io.StringIO()
        with patch.object(sys, "argv", argv), contextlib.redirect_stdout(output), \
             patch.object(stage1, "call_llm") as api:
            stage1.main()
        self.assertEqual(json.loads(output.getvalue()),
                         {"total_images": 2, "repair_existing": 1, "generate_missing": 1})
        api.assert_not_called()
        self.assertFalse(self.output.exists())
        with patch.object(sys, "argv", argv[:-1] + ["302"]), \
             contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            stage1.main()
        self.assertEqual(error.exception.code, 2)

    def test_invalid_run_controls_fail_before_any_api_call(self):
        for controls in ({"start": -1}, {"end": -1}, {"workers": 0}, {"delay": -1}):
            with self.subTest(controls=controls), patch.object(stage1, "call_llm") as api, \
                 self.assertRaisesRegex(ValueError, "Invalid Stage 1"):
                stage1.run_stage1(self.raw, self.mapping, self.output, **controls)
            api.assert_not_called()

    def test_real_run_cli_checks_ngspice_but_validation_does_not(self):
        argv = ["stage1", "--data-dir", str(self.raw), "--mapping", str(self.mapping)]
        with patch.object(stage1.shutil, "which", return_value=None), \
             patch.object(stage1, "call_llm") as api:
            with patch.object(sys, "argv", argv + ["--validate-only"]), \
                 contextlib.redirect_stdout(io.StringIO()):
                stage1.main()
            with patch.object(sys, "argv", argv + ["--output-dir", str(self.output)]), \
                 contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                stage1.main()
            self.assertEqual(error.exception.code, 2)
            api.assert_not_called()

    def test_fingerprint_requires_contents_and_progress_version(self):
        path = self.raw / "images/img1.png"
        before = content_fingerprint({"image": path}, {"mode": "test"})
        path.touch()
        self.assertEqual(content_fingerprint({"image": path}, {"mode": "test"}), before)
        path.write_bytes(b"")
        with self.assertRaisesRegex(ValueError, "Empty fingerprint input"):
            content_fingerprint({"image": path}, {})
        path.unlink()
        with self.assertRaises(FileNotFoundError):
            content_fingerprint({"image": path}, {})
        progress = self.root / "progress.json"
        self.assertEqual(load_progress(progress, "new")["protocol_version"], "new")
        progress.write_text('{"protocol_version": "old", "completed": {}, "failed": {}}')
        with self.assertRaisesRegex(ValueError, "new output directory"):
            load_progress(progress, "new")


if __name__ == "__main__":
    unittest.main()
