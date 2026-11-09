"""Exercise both routing branches with real Ngspice and scripted LLM responses."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_refinement import llm_client, stage1_refine as stage1, stage2_refine as stage2

try:
    import clip
    from PIL import Image
except ImportError:
    clip = Image = None


def circuit(marker, valid=True):
    resistance = "1k" if valid else ""
    return f"spice\n* {marker}\nV1 1 0 1\nR1 1 0 {resistance}\n.end"


@unittest.skipUnless(shutil.which("ngspice") and clip is not None and Image is not None,
                     "Real Ngspice, CLIP tokenizer and Pillow are required")
class RefinementIntegration(unittest.TestCase):
    def test_original_repair_missing_generation_and_feedback_handoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, first, second = (root / name for name in ("raw", "stage1", "stage2"))
            for folder in ("images", "spice", "captions"):
                (raw / folder).mkdir(parents=True)
            for index in (1, 2, 3):
                Image.new("RGB", (2, 2), color=(index, 0, 0)).save(raw / f"images/img{index}.png")
            original_caption = "Original description of the resistor load."
            raw_bad = circuit("RAW_BAD", valid=False)
            (raw / "spice/spice1.txt").write_text(raw_bad)
            (raw / "spice/spice2.txt").write_text(circuit("RAW_GOOD"))
            for index in (1, 2):
                (raw / f"captions/cap{index}.txt").write_text(original_caption)
            mapping = [{"image": f"images/img{i}.png", "spice": f"spice/spice{i}.txt",
                        "caption": f"captions/cap{i}.txt"} for i in (1, 2)]
            mapping_path = raw / "data_mapping.json"
            mapping_path.write_text(json.dumps(mapping))

            first_requests, second_requests = [], []

            def initial_response(**request):
                first_requests.append(request)
                prompt = request["user_prompt"]
                if "RAW_BAD" in prompt:
                    self.assertIn(raw_bad, prompt)
                    self.assertIn("error", prompt.lower())
                    return f"<NETLIST>{circuit('STAGE1_BAD', valid=False)}</NETLIST>"
                if "RAW_GOOD" in prompt:
                    return f"<NETLIST>{circuit('STAGE1_GOOD')}</NETLIST>"
                self.assertTrue(request["image_path"].endswith("img3.png"))
                return (f"<NETLIST>{circuit('GENERATED')}</NETLIST>"
                        "<DESCRIPTION>A generated resistor load.</DESCRIPTION>")

            def feedback_response(**request):
                second_requests.append(request)
                prompt = request["user_prompt"]
                if "STAGE1_BAD" in prompt:
                    self.assertIn("error", prompt.lower())
                    self.assertEqual(Path(request["image_path"]), first / "images/img1.png")
                    return f"<NETLIST>{circuit('REPAIRED')}</NETLIST>"
                self.assertIn(original_caption, prompt)
                self.assertIn("REPAIRED", prompt)
                return "<DESCRIPTION>A voltage source supplies a resistor load.</DESCRIPTION>"

            # A missed mock must fail before any credential lookup/network request.
            with patch.object(llm_client, "get_api_key", side_effect=AssertionError("No live API allowed")), \
                 patch.object(stage1, "call_llm", side_effect=initial_response), \
                 patch.object(stage2, "call_llm", side_effect=feedback_response), \
                 patch.object(stage2.time, "sleep"):
                initial = stage1.run_stage1(raw, mapping_path, first, workers=1, delay=0)
                self.assertEqual(initial["total"], 3)
                self.assertEqual(initial["both_pass"], 2)
                self.assertEqual(initial["failed"], 0)
                self.assertEqual((first / "captions/cap1.txt").read_text(), original_caption)
                final = stage2.run_stage2(raw, first, second, workers=1, delay=0)
                self.assertEqual(final["total"], 3)
                self.assertEqual(final["both_pass"], 3)
                self.assertEqual(final["failed"], 0)
                stage1.run_stage1(raw, mapping_path, first, workers=1, delay=0)
                stage2.run_stage2(raw, first, second, workers=1, delay=0)

            self.assertEqual(len(first_requests), 3)
            self.assertEqual(len(second_requests), 2)  # One repair and one description request.
            self.assertEqual((raw / "spice/spice1.txt").read_text(), raw_bad)
            self.assertEqual((second / "captions/cap2.txt").read_text(), original_caption)
            progress1 = json.loads((first / "stage1_progress.json").read_text())
            progress2 = json.loads((second / "stage2_progress.json").read_text())
            self.assertEqual([progress1["completed"][f"images/img{i}.png"]["source_mode"]
                              for i in (1, 2, 3)],
                             ["repair_existing", "repair_existing", "generate_missing"])
            self.assertTrue(progress1["completed"]["images/img1.png"]["check"]["compile_errors"])
            self.assertEqual([progress2["completed"][f"images/img{i}.png"]["action"]
                              for i in (1, 2, 3)], ["repaired", "accepted", "accepted"])


if __name__ == "__main__":
    unittest.main()
