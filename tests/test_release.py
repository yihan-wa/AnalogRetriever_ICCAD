import json
import random
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate_table2 import average_recall_at_1, markdown, table_rows
from scripts.validate_release import validate_ids, validate_data
from data_refinement.generate_table1 import build_table
DIRECTIONS = ("I2C", "T2I", "T2C", "C2I", "I2T", "C2T")
try:
    from PIL import Image
except ImportError:
    Image = None


class ReleaseChecks(unittest.TestCase):
    def test_average_uses_all_six_directions(self):
        values = [74.7, 78.2, 75.6, 72.0, 78.5, 72.4]
        results = {d: {'R@1': v} for d, v in zip(DIRECTIONS, values)}
        self.assertAlmostEqual(average_recall_at_1(results), 75.2333333333)
        del results['C2T']
        self.assertIsNone(average_recall_at_1(results))

    def test_disjoint_seeded_split(self):
        ids = list(range(1, 21))
        random.Random(42).shuffle(ids)
        validate_ids(ids[5:], ids[:5], total=20, test_size=5)
        with self.assertRaises(ValueError):
            validate_ids(ids[5:-1] + [ids[0]], ids[:5], total=20, test_size=5)
        with self.assertRaises(ValueError):
            validate_ids(ids[5:], ids[:5][::-1], total=20, test_size=5)

    @unittest.skipIf(Image is None, "Pillow is required for image validation")
    def test_missing_or_blank_modality_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for folder, prefix, ext in [('captions','cap','txt'), ('images','img','jpg'), ('spice','spice','txt')]:
                (root / folder).mkdir()
                (root / folder / f'{prefix}1.{ext}').write_bytes(b'x')
            Image.new('RGB', (2, 2)).save(root / 'images/img1.jpg')
            validate_data(root, [1])
            (root / 'captions/cap1.txt').write_text('  ')
            with self.assertRaises(ValueError):
                validate_data(root, [1])
            with self.assertRaises(ValueError):
                validate_data(root, [1, 2])

    def test_table1_rates_and_intersection_come_from_per_file_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def write(name, flags):
                (root / name).write_text(json.dumps({"details": [
                    {"compile_success": c, "dc_pass": d} for c, d in flags]}))
            write('original.json', [(True, False), (False, False)])
            write('stage1.json', [(True, True), (True, False), (False, False)])
            write('stage2.json', [(True, True), (True, False), (False, True)])
            rows = build_table(root / 'original.json', root / 'stage1.json', root / 'stage2.json')
        self.assertEqual([row['stage'] for row in rows],
                         ['Original Masala-CHAI', '+ Stage 1 (Initial Repair)',
                          '+ Stage 2 (Feedback Refinement)', 'Passing both checks'])
        self.assertEqual(rows[0], {'stage': 'Original Masala-CHAI', 'triplets': 2, 'compile': 50.0, 'dc_pass': 0.0})
        self.assertEqual(rows[2]['compile'], 66.67)
        self.assertEqual(rows[3]['triplets'], 1)

    def test_table2_rows_follow_paper_order_and_mark_missing_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'analogretriever.json').write_text(json.dumps(
                {d: {'R@1': 75.0, 'R@5': 95.0, 'R@10': 98.0} for d in DIRECTIONS}))
            rows = table_rows(root)
        self.assertEqual([model for model, _, _ in rows],
                         ['CLIP', 'CROP', 'ChatLS', 'NetTAG', 'TI (Bi-Modal)', 'TIC (GCN)',
                          'TIC (RGCN)', 'AnalogRetriever'])
        self.assertIsNone(rows[0][1])
        self.assertEqual(rows[-1][2][-1], '75.0')
        self.assertIn('| AnalogRetriever | 75.0 | 95.0 | 98.0', markdown(rows))


if __name__ == '__main__':
    unittest.main()
