"""Curriculum sampler, schedules, losses and optimizer parameter registration."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import numpy as np
    import torch
    import torch.nn as nn
    from analogretriever.training.common import (accumulation_group_size, clip_parameters,
                                                 code_alignment_loss, contrastive_loss,
                                                 optimizer_updates_per_epoch, trimodal_loss,
                                                 warmup_cosine_schedule)
    from analogretriever.training.train_curriculum import (CurriculumBatchSampler, curriculum_alpha,
                                                           load_cluster_labels)
except ImportError:
    CurriculumBatchSampler = None


def fake_model():
    """Image and text wrappers that share one module, as CLIP does, plus a netlist encoder."""
    shared = nn.ModuleDict({"visual": nn.Linear(2, 2), "text": nn.Linear(2, 2)})
    shared["visual"].bias.requires_grad = False
    model = nn.Module()
    model.image_encoder = nn.Module()
    model.image_encoder.model = shared
    model.text_encoder = nn.Module()
    model.text_encoder.model = shared
    model.netlist_encoder = nn.Linear(2, 2)
    return model


@unittest.skipIf(CurriculumBatchSampler is None, "Training dependencies are required")
class CurriculumChecks(unittest.TestCase):
    def test_alpha_schedule_matches_eq6(self):
        alphas = [curriculum_alpha(m, 12, 0.05, 0.3) for m in range(1, 13)]
        self.assertAlmostEqual(alphas[0], 0.05)
        self.assertAlmostEqual(alphas[-1], 0.3)
        self.assertTrue(np.allclose(np.diff(alphas), 0.25 / 11))

    def test_batches_have_seed_cluster_fraction_and_no_duplicates(self):
        labels = [i % 4 for i in range(80)]
        for alpha in (0.0, 0.3):
            sampler = CurriculumBatchSampler(labels, batch_size=20, alpha=alpha)
            np.random.seed(0)
            with patch("analogretriever.training.train_curriculum.random.shuffle"):
                batches = list(sampler)
            self.assertEqual(len(batches), 4)
            for batch in batches:
                seed_cluster = labels[batch[0]]
                self.assertEqual(len(set(batch)), 20)
                self.assertEqual(sum(labels[i] == seed_cluster for i in batch), 1 + int(20 * alpha))

    def test_small_clusters_fall_back_to_unused_circuits(self):
        sampler = CurriculumBatchSampler([0, 0, 0, 1], batch_size=4, alpha=0.75)
        batch = next(iter(sampler))
        self.assertEqual(sorted(batch), [0, 1, 2, 3])

    def test_invalid_alpha_is_rejected(self):
        sampler = CurriculumBatchSampler([0, 1], batch_size=1)
        for alpha in (-0.1, 1.1, float("nan")):
            with self.assertRaises(ValueError):
                sampler.set_alpha(alpha)

    def test_cluster_labels_follow_csv_ids_and_missing_ids_fail(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "clusters.json"
            path.write_text(json.dumps({"31": 1, "12": 0}))
            self.assertEqual(load_cluster_labels(path, ["12", "31"]), [0, 1])
            path.write_text(json.dumps({"12": 0}))
            with self.assertRaisesRegex(ValueError, "no cluster"):
                load_cluster_labels(path, ["12", "31"])


@unittest.skipIf(CurriculumBatchSampler is None, "Training dependencies are required")
class OptimizationChecks(unittest.TestCase):
    def test_gradient_accumulation_groups_include_short_tail(self):
        self.assertEqual(optimizer_updates_per_epoch(10, 4), 3)
        self.assertEqual([accumulation_group_size(s, 10, 4) for s in range(10)], [4] * 8 + [2] * 2)
        with self.assertRaises(ValueError):
            optimizer_updates_per_epoch(0, 1)

    def test_standard_mode_registers_shared_clip_parameters_once(self):
        model = fake_model()
        parameters = clip_parameters(model, "standard", "cpu")
        self.assertEqual(len(parameters), 3)
        self.assertEqual(len({id(p) for p in parameters}), 3)
        self.assertFalse(any(p is model.netlist_encoder.weight for p in parameters))

    def test_legacy_mode_registers_shared_parameters_twice_and_requires_cuda(self):
        model = fake_model()
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            clip_parameters(model, "legacy_cuda_duplicate", "cpu")
        if torch.cuda.is_available():
            parameters = clip_parameters(model, "legacy_cuda_duplicate", "cuda")
            self.assertEqual(len(parameters), 6)
            self.assertEqual(len({id(p) for p in parameters}), 3)
        with self.assertRaises(ValueError):
            clip_parameters(model, "unknown", "cpu")

    def test_trimodal_loss_sums_three_symmetric_pairs(self):
        torch.manual_seed(0)
        v_i, v_t, v_c = (torch.nn.functional.normalize(torch.randn(6, 4), dim=1) for _ in range(3))
        scale = torch.tensor(np.log(1 / 0.07))
        expected = sum(contrastive_loss(a, b, scale, 0.1) for a, b in ((v_i, v_t), (v_i, v_c), (v_t, v_c)))
        self.assertTrue(torch.isclose(trimodal_loss(v_i, v_t, v_c, scale, 0.1), expected))
        self.assertTrue(torch.isclose(contrastive_loss(v_i, v_t, scale, 0.0), contrastive_loss(v_t, v_i, scale, 0.0)))
        self.assertTrue(torch.isclose(code_alignment_loss(v_i, v_t, v_c, scale, 0.1),
                                      contrastive_loss(v_i, v_c, scale, 0.1) + contrastive_loss(v_t, v_c, scale, 0.1)))

    def test_warmup_cosine_schedule(self):
        optimizer = torch.optim.SGD([nn.Parameter(torch.zeros(1))], lr=1.0)
        scheduler = warmup_cosine_schedule(optimizer, total_steps=10, warmup_steps=2)
        rates = []
        for _ in range(11):
            rates.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            scheduler.step()
        self.assertEqual(rates[:3], [0.0, 0.5, 1.0])
        self.assertAlmostEqual(rates[-1], 0.01)
        self.assertAlmostEqual(min(rates[2:]), 0.01)


if __name__ == "__main__":
    unittest.main()
