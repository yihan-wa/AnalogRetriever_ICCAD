"""Zero-shot baselines: netlist representations, caches and pooling."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
try:
    import torch
    from analogretriever.evaluation import evaluate_chatls as chatls
    from analogretriever.evaluation import evaluate_nettag as nettag
    from analogretriever.evaluation import zero_shot
except ImportError:
    torch = None

RECIPE = {"prompt": "p", "model": "m"}


@unittest.skipIf(torch is None, "Baseline dependencies are required")
class BaselineChecks(unittest.TestCase):
    def test_summary_cache_requires_matching_inputs_recipe_and_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "summaries.json"
            fingerprint = zero_shot.content_fingerprint(["netlist A", "netlist B"], RECIPE)
            zero_shot.save_summaries(path, fingerprint, ["summary A", "summary B"], RECIPE)
            self.assertEqual(zero_shot.load_summaries(path, fingerprint, 2), ["summary A", "summary B"])
            for inputs, recipe in ((["netlist C", "netlist B"], RECIPE), (["netlist B", "netlist A"], RECIPE),
                                   (["netlist A", "netlist B"], {**RECIPE, "model": "other"})):
                self.assertIsNone(zero_shot.load_summaries(path, zero_shot.content_fingerprint(inputs, recipe), 2))
            self.assertIsNone(zero_shot.load_summaries(path, fingerprint, 1))
            path.write_text('["legacy summary"]')
            self.assertIsNone(zero_shot.load_summaries(path, fingerprint, 1))
            path.write_text("{invalid JSON")
            self.assertIsNone(zero_shot.load_summaries(path, fingerprint, 2))
            with self.assertRaises(ValueError):
                zero_shot.save_summaries(path, fingerprint, ["", "x"], RECIPE)

    def test_nettag_keeps_every_device_and_its_parameters(self):
        attributes = nettag.device_attributes("\n".join(f"R{i} {i} 0 {i + 1}k" for i in range(60)))
        self.assertEqual(len(attributes), 60)
        self.assertIn("[Parameters] 60k", attributes[-1])
        source = ("Q1 c b e NPNPWR\nQ2 c2 b2 e2 body qpart AREA=2\nM1 d g s bulk n_named W=2u L=.2u\n"
                  "K1 L1 L2 0.9\n.model NPNPWR NPN\n.model qpart\n+ PNP (BF=100)\n.model n_named PMOS\n")
        attributes = nettag.device_attributes(source)
        self.assertIn("[Type] NPN", attributes[0])
        self.assertIn("bulk=BODY", attributes[1])
        self.assertIn("[Type] PMOS", attributes[2])
        self.assertIn("[Type] MUTUAL_INDUCTANCE", attributes[3])
        analysis = chatls.structural_analysis(source)
        self.assertIn("4 devices", analysis)
        for kind in ("NPN", "PNP", "PMOS"):
            self.assertIn(f"1 {kind}", analysis)

    def test_device_embeddings_are_mean_pooled_per_circuit(self):
        vectors = torch.tensor([[3., 4.], [4., 0.], [0., 2.], [-1., 0.]])

        class Batch(dict):
            def to(self, device):
                return self

        class Tokenizer:
            queries = []

            def __call__(self, texts, **kwargs):
                queries = [text.split("Query: ", 1)[1] for text in texts]
                self.queries.extend(queries)
                ids = torch.tensor([[ord(query) - ord("a")] for query in queries])
                return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))

        class Model(torch.nn.Module):
            def forward(self, input_ids, attention_mask):
                return SimpleNamespace(last_hidden_state=vectors[input_ids])

        tokenizer = Tokenizer()
        with contextlib.redirect_stderr(io.StringIO()):
            actual = nettag.embed_circuits([["a", "b", "c"], ["d"]], "cpu", batch_size=2,
                                           tokenizer=tokenizer, model=Model())
        normalized = torch.nn.functional.normalize(vectors, dim=1)
        expected = torch.nn.functional.normalize(torch.stack([normalized[:3].mean(0), normalized[3]]), dim=1)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(tokenizer.queries, ["a", "b", "c", "d"])

    def test_projection_is_fixed_and_leaves_global_rng_untouched(self):
        embeddings = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
        torch.manual_seed(123)
        before = torch.get_rng_state().clone()
        first = nettag.project(embeddings, 2)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        torch.manual_seed(789)
        torch.testing.assert_close(first, nettag.project(embeddings, 2))

    def test_summary_generation_uses_greedy_decoding(self):
        class Batch(dict):
            @property
            def input_ids(self):
                return self["input_ids"]

            def to(self, device):
                return self

        tokenizer = mock.Mock()
        tokenizer.apply_chat_template.return_value = "prompt"
        tokenizer.return_value = Batch(input_ids=torch.tensor([[1, 2]]))
        tokenizer.decode.return_value = "A test summary."
        model = mock.Mock()
        model.eval.return_value = model
        model.generate.return_value = torch.tensor([[1, 2, 3]])
        transformers = SimpleNamespace(
            AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=tokenizer)),
            AutoModelForCausalLM=SimpleNamespace(from_pretrained=mock.Mock(return_value=model)),
        )
        with mock.patch.dict(sys.modules, {"transformers": transformers}), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(zero_shot.generate_summaries(["R1 1 0 1k"], 80, "cpu"), ["A test summary."])
        call = model.generate.call_args
        self.assertIs(call.kwargs["do_sample"], False)
        self.assertNotIn("temperature", call.kwargs)


if __name__ == "__main__":
    unittest.main()
