"""Stage 2 protocol regressions without paid model calls or model downloads."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_refinement import stage2_refine as stage2
from data_refinement.pipeline_io import output_fingerprint


NETLIST = "spice\nV1 1 0 1\nR1 1 0 1k\n.end\n"
REPAIRED = NETLIST.replace("1k", "2k")
CAPTION = "Original caption from the raw dataset."
GOOD = {"compile_success": True, "dc_pass": True, "compile_errors": "", "dc_errors": ""}
BAD = {"compile_success": True, "dc_pass": False, "dc_errors": "No operating point",
       "dc_output": "Full simulator output: singular matrix"}


class Stage2ProtocolChecks(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw, self.first, self.second = [self.root / name for name in ("raw", "stage1", "stage2")]
        self.raw.mkdir()
        for folder in ("images", "spice", "captions"):
            (self.first / folder).mkdir(parents=True)
        self.entry = {"image": "images/img1.jpg", "spice": "spice/spice1.txt", "caption": "captions/cap1.txt"}
        (self.first / self.entry["image"]).write_bytes(b"Stage 1 source image")
        (self.first / self.entry["spice"]).write_text(NETLIST)
        (self.first / self.entry["caption"]).write_text(CAPTION)
        (self.first / "data_mapping.json").write_text(json.dumps([self.entry]))
        self.record = {"success": True, "image": self.entry["image"],
                       "output_spice": self.entry["spice"], "output_caption": self.entry["caption"],
                       "source_mode": "repair_existing", "original_spice": "spice/raw1.txt",
                       "original_caption": "captions/raw1.txt", "input_fingerprint": "source-v1"}
        self.update_stage1()

    def update_stage1(self):
        self.record["output_fingerprint"] = output_fingerprint(self.first, self.record)
        progress = {"protocol_version": stage2.STAGE1_PROTOCOL_VERSION,
                    "completed": {self.entry["image"]: self.record}, "failed": {}}
        (self.first / "stage1_progress.json").write_text(json.dumps(progress))

    def run_second(self, **kwargs):
        return stage2.run_stage2(self.raw, self.first, self.second, workers=1, delay=0, **kwargs)

    def result(self):
        progress = json.loads((self.second / "stage2_progress.json").read_text())
        return progress["completed"][self.entry["image"]]

    def repaired_result(self):
        return {**GOOD, "netlist": REPAIRED, "iterations": 1,
                "iteration_log": [{"iteration": 0, "netlist": NETLIST, "check": BAD},
                                  {"iteration": 1, "netlist": REPAIRED, "check": GOOD}]}

    def test_passing_stage1_is_preserved_without_any_llm_request(self):
        with patch.object(stage2, "check_netlist", return_value=GOOD), \
             patch.object(stage2, "repair_netlist") as repair, \
             patch.object(stage2, "refine_description") as refine, \
             patch.object(stage2, "call_llm") as api:
            self.assertEqual(self.run_second()["both_pass"], 1)
        repair.assert_not_called()
        refine.assert_not_called()
        api.assert_not_called()
        self.assertEqual((self.second / self.entry["spice"]).read_text(), NETLIST)
        self.assertEqual((self.second / self.entry["caption"]).read_text(), CAPTION)
        self.assertEqual((self.second / self.entry["image"]).read_bytes(), b"Stage 1 source image")
        self.assertEqual(self.result()["action"], "accepted")
        self.assertFalse(self.result()["description_refined"])
        mapping = json.loads((self.second / "data_mapping.json").read_text())
        self.assertEqual(set(mapping[0]), {"image", "spice", "caption"})

    def test_failed_stage1_uses_original_caption_after_verified_repair(self):
        with patch.object(stage2, "check_netlist", return_value=BAD), \
             patch.object(stage2, "repair_netlist", return_value=self.repaired_result()) as repair, \
             patch.object(stage2, "refine_description", return_value="Refined caption.") as refine:
            self.assertEqual(self.run_second()["both_pass"], 1)
        repair.assert_called_once_with(NETLIST, str(self.first / self.entry["image"]), 5, initial_check=BAD)
        refine.assert_called_once_with(REPAIRED, CAPTION)
        result = self.result()
        self.assertEqual(result["action"], "repaired")
        self.assertEqual(result["initial_check"], BAD)
        self.assertEqual(result["original_caption"], "captions/raw1.txt")
        self.assertEqual(len(result["iteration_log"]), 2)

    def test_unverified_repair_retains_caption_and_reports_quality_failure(self):
        failed = {**BAD, "netlist": REPAIRED, "iterations": 1, "iteration_log": []}
        with patch.object(stage2, "check_netlist", return_value=BAD), \
             patch.object(stage2, "repair_netlist", return_value=failed), \
             patch.object(stage2, "refine_description") as refine:
            summary = self.run_second(max_iters=1)
        refine.assert_not_called()
        self.assertEqual(summary["both_pass"], 0)
        self.assertEqual(summary["failed"], 0)  # A complete attempt, not an API failure.
        self.assertEqual(self.result()["action"], "quality_failed")
        self.assertEqual((self.second / self.entry["caption"]).read_text(), CAPTION)

    def test_resume_and_upstream_or_prompt_changes(self):
        with patch.object(stage2, "check_netlist", return_value=BAD), \
             patch.object(stage2, "repair_netlist", return_value=self.repaired_result()) as repair, \
             patch.object(stage2, "refine_description", return_value="Refined caption.") as refine:
            self.run_second()
            self.run_second()
            self.assertEqual(repair.call_count, 1)
            (self.first / self.entry["caption"]).write_text("Corrected original caption.")
            self.record["input_fingerprint"] = "source-v2"
            self.update_stage1()
            self.run_second()
            refine.assert_called_with(REPAIRED, "Corrected original caption.")
            self.assertEqual(repair.call_count, 2)
            with patch.object(stage2, "REPAIR_SYSTEM_PROMPT", stage2.REPAIR_SYSTEM_PROMPT + " Updated."):
                self.run_second()
            self.assertEqual(repair.call_count, 3)

    def test_changed_stage2_output_is_not_reused(self):
        with patch.object(stage2, "check_netlist", return_value=GOOD) as check:
            self.run_second()
            (self.second / self.entry["caption"]).write_text("Tampered output")
            self.run_second()
            self.assertEqual(check.call_count, 2)
        self.assertEqual((self.second / self.entry["caption"]).read_text(), CAPTION)

    def test_changed_upstream_without_provenance_update_fails_before_api(self):
        (self.first / self.entry["spice"]).write_text(REPAIRED)
        with patch.object(stage2, "call_llm") as api:
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.run_second()
        api.assert_not_called()

    def test_legacy_progress_is_rejected_for_both_stages(self):
        with patch.object(stage2, "call_llm") as api:
            (self.first / "stage1_progress.json").write_text(json.dumps({"completed": {}}))
            with self.assertRaises(ValueError):
                self.run_second()
            self.update_stage1()
            self.second.mkdir()
            (self.second / "stage2_progress.json").write_text(json.dumps({"completed": {}}))
            with self.assertRaises(ValueError):
                self.run_second()
        api.assert_not_called()

    def test_missing_upstream_file_fails_before_any_model_call(self):
        (self.first / self.entry["caption"]).unlink()
        with patch.object(stage2, "call_llm") as api:
            with self.assertRaisesRegex(ValueError, "Incomplete Stage 1"):
                self.run_second()
        api.assert_not_called()

    def test_duplicate_mapping_is_rejected_before_model_calls(self):
        (self.first / "data_mapping.json").write_text(json.dumps([self.entry, self.entry]))
        with patch.object(stage2, "call_llm") as api:
            with self.assertRaisesRegex(ValueError, "Duplicate Stage 1"):
                self.run_second()
        api.assert_not_called()

    def test_interrupted_stage1_export_cannot_omit_completed_samples(self):
        second_entry = {"image": "images/img2.jpg", "spice": "spice/spice2.txt",
                        "caption": "captions/cap2.txt"}
        for field, target in second_entry.items():
            (self.first / target).write_bytes((self.first / self.entry[field]).read_bytes())
        second_record = {**self.record, "image": second_entry["image"],
                         "output_spice": second_entry["spice"], "output_caption": second_entry["caption"]}
        second_record["output_fingerprint"] = output_fingerprint(self.first, second_record)
        progress_path = self.first / "stage1_progress.json"
        progress = json.loads(progress_path.read_text())
        progress["completed"][second_entry["image"]] = second_record
        progress_path.write_text(json.dumps(progress))
        # A crash after the new progress write leaves the old one-entry mapping.
        with patch.object(stage2, "call_llm") as api, patch.object(stage2, "check_netlist") as check:
            with self.assertRaisesRegex(ValueError, "mapping/progress mismatch.*rerun Stage 1"):
                self.run_second()
        api.assert_not_called()
        check.assert_not_called()
        # Finishing Stage 1's export restores the exact correspondence.
        (self.first / "data_mapping.json").write_text(json.dumps([self.entry, second_entry]))
        mapping, records = stage2._stage1_inputs(self.first)
        self.assertEqual(len(mapping), 2)
        self.assertEqual(set(records), {self.entry["image"], second_entry["image"]})

    def test_completed_record_image_must_match_its_mapping_key(self):
        other_image = "images/img2.jpg"
        (self.first / other_image).write_bytes((self.first / self.entry["image"]).read_bytes())
        self.record["image"] = other_image
        self.update_stage1()  # Valid output hash, but bound to the wrong image.
        with patch.object(stage2, "call_llm") as api:
            with self.assertRaisesRegex(ValueError, "provenance mismatch"):
                self.run_second()
        api.assert_not_called()

    def test_repair_preserves_full_feedback_and_final_check(self):
        second_bad = {"compile_success": False, "dc_pass": False,
                      "compile_errors": "Unknown model", "raw_output": "Full compiler transcript"}
        with patch.object(stage2, "check_netlist", side_effect=[BAD, second_bad, GOOD]), \
             patch.object(stage2, "call_llm", return_value=f"<NETLIST>{REPAIRED}</NETLIST>") as api, \
             patch.object(stage2.time, "sleep"):
            result = stage2.repair_netlist(NETLIST, "source.jpg", max_iters=2)
        self.assertEqual(api.call_count, 2)
        second_prompt = api.call_args_list[1].kwargs["user_prompt"]
        self.assertIn("Full compiler transcript", second_prompt)
        self.assertIn("Full simulator output: singular matrix", second_prompt)
        self.assertEqual(result["iterations"], 2)
        self.assertEqual([item["check"] for item in result["iteration_log"]], [BAD, second_bad, GOOD])

    def test_repair_api_failure_is_retryable_and_never_completed(self):
        with patch.object(stage2, "check_netlist", side_effect=[BAD, BAD, GOOD]), \
             patch.object(stage2, "call_llm", side_effect=[None, f"<NETLIST>{REPAIRED}</NETLIST>"]) as api, \
             patch.object(stage2, "refine_description", return_value="Refined caption."), \
             patch.object(stage2.time, "sleep"):
            self.assertEqual(self.run_second(max_iters=1)["failed"], 1)
            self.assertEqual(json.loads((self.second / "data_mapping.json").read_text()), [])
            self.assertEqual(self.run_second(max_iters=1)["failed"], 0)
        self.assertEqual(api.call_count, 2)
        self.assertEqual(self.result()["action"], "repaired")

    def test_zero_repair_budget_still_checks_and_reports_failure(self):
        with patch.object(stage2, "check_netlist", return_value=BAD), \
             patch.object(stage2, "call_llm") as api:
            result = stage2.repair_netlist(NETLIST, "source.jpg", max_iters=0)
        api.assert_not_called()
        self.assertEqual(result["iterations"], 0)
        self.assertEqual(len(result["iteration_log"]), 1)
        self.assertFalse(result["dc_pass"])


if __name__ == "__main__":
    unittest.main()
