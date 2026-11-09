"""Fixed retrieval protocol of paper Sec. 4.2."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
    from analogretriever.evaluation.protocol import (DIRECTIONS, fixed_protocol_recall, gallery_indices,
                                                     mean_recall_at_1, recall_at_k)
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required")
class ProtocolChecks(unittest.TestCase):
    def test_gallery_draws_use_fixed_seeds_without_touching_global_rng(self):
        torch.manual_seed(5)
        before = torch.get_rng_state().clone()
        draws = [gallery_indices(1000, draw) for draw in range(10)]
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        expected = torch.randperm(1000, generator=torch.Generator().manual_seed(3 * 42 + 7))[:200]
        self.assertTrue(torch.equal(draws[3], expected))
        self.assertTrue(all(len(set(d.tolist())) == 200 for d in draws))

    def test_recall_at_k(self):
        gallery = torch.eye(4)
        queries = torch.tensor([[1., 0, 0, 0], [0, 0, 1., 0], [0, 0, 1., 0], [0, 0, 0, 1.]])
        self.assertEqual(recall_at_k(queries, gallery, ks=(1,)), {"R@1": 75.0})

    def test_identical_embeddings_give_perfect_recall_and_all_directions(self):
        torch.manual_seed(0)
        embeddings = torch.nn.functional.normalize(torch.randn(300, 16), dim=1)
        results = fixed_protocol_recall({"I": embeddings, "T": embeddings, "C": embeddings})
        self.assertEqual(list(results), list(DIRECTIONS))
        self.assertEqual(mean_recall_at_1(results), 100.0)
        bimodal = fixed_protocol_recall({"I": embeddings, "T": embeddings})
        self.assertEqual(list(bimodal), ["T2I", "I2T"])

    def test_mean_uses_all_six_directions(self):
        values = [74.7, 78.2, 75.6, 72.0, 78.5, 72.4]
        results = {name: {"R@1": value} for name, value in zip(DIRECTIONS, values)}
        self.assertAlmostEqual(mean_recall_at_1(results), 75.2333333333)


if __name__ == "__main__":
    unittest.main()
