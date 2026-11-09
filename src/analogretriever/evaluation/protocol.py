"""Fixed retrieval protocol of paper Sec. 4.2.

Each draw samples a gallery of 200 items from the 1,000 held-out triplets;
the same items serve as queries and candidates. Recall@K is averaged over ten
draws with fixed seeds (random-guess R@1 is 0.5%).
"""
import json
from pathlib import Path

import torch

# Direction name -> (query modality, gallery modality); I: image, T: text, C: netlist.
DIRECTIONS = {
    "I2C": ("I", "C"), "T2I": ("T", "I"), "T2C": ("T", "C"),
    "C2I": ("C", "I"), "I2T": ("I", "T"), "C2T": ("C", "T"),
}
DIRECTION_LABELS = {
    "I2C": "Image -> Code", "T2I": "Text -> Image", "T2C": "Text -> Code",
    "C2I": "Code -> Image", "I2T": "Image -> Text", "C2T": "Code -> Text",
}
GALLERY_SIZE = 200
NUM_DRAWS = 10
RECALL_KS = (1, 5, 10)
# Checkpoint selection metric: mean R@1 of these directions (T2I alone for TI).
SELECTION_DIRECTIONS = ("I2C", "T2I", "T2C")


def gallery_indices(num_items, draw, size=GALLERY_SIZE):
    generator = torch.Generator().manual_seed(draw * 42 + 7)
    return torch.randperm(num_items, generator=generator)[:size]


def recall_at_k(queries, gallery, ks=RECALL_KS):
    """Recall@K (%) where query i matches gallery item i, ranked by cosine similarity."""
    similarity = queries @ gallery.t()
    _, topk = similarity.topk(min(max(ks), gallery.shape[0]), dim=1, largest=True, sorted=True)
    target = torch.arange(queries.shape[0], device=queries.device).view(-1, 1)
    return {f"R@{k}": (topk[:, :k] == target).any(dim=1).float().mean().item() * 100.0 for k in ks}


def fixed_protocol_recall(embeddings, directions=None, num_draws=NUM_DRAWS, gallery_size=GALLERY_SIZE):
    """Average Recall@K over the fixed gallery draws.

    ``embeddings`` maps modality letters ("I", "T", "C") to row-aligned,
    L2-normalized tensors. Directions default to all those whose two
    modalities are present.
    """
    if directions is None:
        directions = [name for name, (query, gallery) in DIRECTIONS.items()
                      if query in embeddings and gallery in embeddings]
    num_items = len(next(iter(embeddings.values())))
    results = {}
    for name in directions:
        query, gallery = DIRECTIONS[name]
        totals = dict.fromkeys((f"R@{k}" for k in RECALL_KS), 0.0)
        for draw in range(num_draws):
            indices = gallery_indices(num_items, draw, gallery_size)
            for key, value in recall_at_k(embeddings[query][indices], embeddings[gallery][indices]).items():
                totals[key] += value
        results[name] = {key: value / num_draws for key, value in totals.items()}
    return results


def mean_recall_at_1(results, directions=tuple(DIRECTIONS)):
    return sum(results[name]["R@1"] for name in directions) / len(directions)


def format_results(results):
    lines = [f"  {DIRECTION_LABELS[name]:<14} " + "  ".join(f"{key}={value:5.1f}%" for key, value in metrics.items())
             for name, metrics in results.items() if name in DIRECTIONS]
    if all(name in results for name in DIRECTIONS):
        lines.append(f"  Average R@1 over six directions: {mean_recall_at_1(results):.1f}%")
    return "\n".join(lines)


def save_results(results, path, metadata=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(results)
    if metadata is not None:
        payload["_metadata"] = metadata
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
