"""Checkpoint sidecars, model-type inference and graph-schema checks."""
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
    from analogretriever.checkpoints import (_check_graph_compatibility, infer_model_type, metadata_path,
                                             save_checkpoint)
    from analogretriever.dataset import NODE_TYPES, RELATION_TYPES
    from analogretriever.encoders import NetlistEncoder
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Model dependencies are required")
class CheckpointChecks(unittest.TestCase):
    def state(self, conv):
        encoder = NetlistEncoder(conv=conv, hidden_dim=8, out_dim=4)
        return {"netlist_encoder." + key: value for key, value in encoder.state_dict().items()}

    def test_model_type_is_inferred_from_weights(self):
        self.assertEqual(infer_model_type({"image_encoder.model.x": torch.zeros(1)}), "TI")
        self.assertEqual(infer_model_type(self.state("rgcn")), "TIC (RGCN)")
        self.assertEqual(infer_model_type(self.state("gcn")), "TIC (GCN)")

    def test_relation_and_node_type_counts_are_checked(self):
        state = self.state("rgcn")
        _check_graph_compatibility(state, None)
        self.assertEqual(state["netlist_encoder.conv1.weight"].shape[0], len(RELATION_TYPES))
        wrong = dict(state, **{"netlist_encoder.conv1.weight": torch.zeros(22, 8, 8)})
        with self.assertRaisesRegex(ValueError, "22 relation types"):
            _check_graph_compatibility(wrong, None)
        wrong = dict(state, **{"netlist_encoder.type_emb.weight": torch.zeros(len(NODE_TYPES) + 1, 8)})
        with self.assertRaisesRegex(ValueError, "node types"):
            _check_graph_compatibility(wrong, None)
        with self.assertRaisesRegex(ValueError, "graph schema"):
            _check_graph_compatibility(state, {"graph": {"schema_version": 1, "node_types": ["X"],
                                                         "relation_types": list(RELATION_TYPES)}})

    def test_sidecar_records_model_graph_and_hash(self):
        model = torch.nn.Module()
        model.netlist_encoder = NetlistEncoder(hidden_dim=8, out_dim=4)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pth"
            save_checkpoint(model, torch.tensor(2.0), path, model_type="AnalogRetriever",
                            training_config={"seed": 42}, selection={"epoch": 3})
            metadata = json.loads(metadata_path(path).read_text())
            state = torch.load(path, weights_only=True)
        self.assertEqual(metadata["model_type"], "AnalogRetriever")
        self.assertEqual(metadata["graph"]["relation_types"], list(RELATION_TYPES))
        self.assertEqual(len(metadata["checkpoint_sha256"]), 64)
        self.assertEqual(state["logit_scale"].item(), 2.0)
        with self.assertRaises(ValueError):
            save_checkpoint(model, torch.tensor(1.0), path, model_type="unknown",
                            training_config={}, selection={})

    def test_encoder_forward_and_relation_gradients(self):
        torch.manual_seed(0)
        encoder = NetlistEncoder(hidden_dim=16, out_dim=8)
        types = torch.tensor([3, 0, 7, 0])
        edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
        relations = torch.tensor([12, 12, 16, 16])
        output = encoder(types, torch.rand(4, 4), edges, torch.tensor([0, 0, 1, 1]), relations)
        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertTrue(torch.allclose(output.norm(dim=1), torch.ones(2)))
        output[:, 0].sum().backward()
        for relation in (12, 16):
            self.assertGreater(encoder.conv1.weight.grad[relation].abs().sum().item(), 0.0)


if __name__ == "__main__":
    unittest.main()
