"""Graph/token cache invalidation of TriModalCircuitDataset on tiny inputs."""
import functools
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
try:
    import torch
    from PIL import Image, UnidentifiedImageError
    from analogretriever.dataset import TriModalCircuitDataset
except ImportError:
    TriModalCircuitDataset = None


def tokenize(text, truncate=True, scale=1):
    return torch.tensor([[len(text[0]) * scale]])


def image_transform(image):
    return torch.full((3, 2, 2), float(image.getpixel((0, 0))[0]))


@unittest.skipIf(TriModalCircuitDataset is None, 'Dataset dependencies are required')
class DatasetCacheChecks(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.csv = self.root / 'sample.csv'
        self.netlist = self.root / 'sample.cir'
        self.picture = self.root / 'sample.png'
        self.netlist.write_text('R1 OUT 0 1k\n')
        self.csv.write_text('id,image,spice,caption\n1,sample.png,sample.cir,first\n')
        Image.new('RGB', (2, 2), color=(10, 0, 0)).save(self.picture)

    def build(self, **kwargs):
        return TriModalCircuitDataset(
            str(self.root), str(self.csv), kwargs.pop('tokenizer', tokenize),
            kwargs.pop('image_preprocess', image_transform), **kwargs)

    def test_unchanged_inputs_reuse_cache(self):
        first = self.build()
        with patch.object(TriModalCircuitDataset, 'process', side_effect=AssertionError('rebuilt')):
            second = self.build()
        self.assertTrue(first[0].x_continuous.equal(second[0].x_continuous))

    def test_netlist_content_change_invalidates_cache_even_with_same_mtime(self):
        first = self.build()
        self.assertEqual(first[0].x_continuous[0, 2].item(), 1000)
        previous = self.netlist.stat()
        self.netlist.write_text('R1 OUT 0 2k\n')
        os.utime(self.netlist, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        second = self.build()
        self.assertEqual(second[0].x_continuous[0, 2].item(), 2000)

    def test_caption_change_rebuilds_cached_tokens(self):
        first = self.build()
        self.assertEqual(first[0].text_tokens.item(), 5)
        self.csv.write_text('id,image,spice,caption\n1,sample.png,sample.cir,longer caption\n')
        second = self.build()
        self.assertEqual(second[0].text_tokens.item(), len('longer caption'))

    def test_changed_tokenizer_configuration_rebuilds_tokens(self):
        self.build(tokenizer=functools.partial(tokenize, scale=1))
        second = self.build(tokenizer=functools.partial(tokenize, scale=2))
        self.assertEqual(second[0].text_tokens.item(), 10)

    def test_image_content_is_part_of_the_fingerprint(self):
        first = self.build()
        before = first._input_fingerprint()
        Image.new('RGB', (2, 2), color=(20, 0, 0)).save(self.picture)
        second = self.build()
        self.assertNotEqual(before, second._input_fingerprint())
        self.assertEqual(second[0].image_tensor[0, 0, 0, 0].item(), 20)

    def test_random_image_transform_runs_on_every_access(self):
        calls = []

        def varying_transform(image):
            calls.append(1)
            return torch.full((3, 2, 2), len(calls))

        dataset = self.build(image_preprocess=varying_transform)
        self.assertEqual(dataset[0].image_tensor[0, 0, 0, 0].item(), 1)
        self.assertEqual(dataset[0].image_tensor[0, 0, 0, 0].item(), 2)

    def test_missing_netlist_and_invalid_graph_are_not_silently_skipped(self):
        self.netlist.unlink()
        with self.assertRaises(FileNotFoundError):
            self.build()
        self.netlist.write_text('F1 out 0 Vmissing 10\n')
        with self.assertRaisesRegex(ValueError, 'Failed to preprocess CSV row 0'):
            self.build()

    def test_corrupt_image_is_not_replaced_with_a_black_image(self):
        self.picture.write_bytes(b'not an image')
        dataset = self.build()
        with self.assertRaises(UnidentifiedImageError):
            dataset[0]


if __name__ == '__main__':
    unittest.main()
