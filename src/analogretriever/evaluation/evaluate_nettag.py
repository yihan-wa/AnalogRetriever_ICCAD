"""NetTAG baseline of paper Table 2: per-device text attributes encoded with
Qwen3-Embedding and pooled into a circuit embedding.

Each device becomes a text attribute with its name, type, terminal
connections and parameters. Qwen3-Embedding-0.6B encodes every device
separately; the normalized device embeddings are mean-pooled per circuit.
A fixed seeded Gaussian projection maps the 1024-d result to the 768-d CLIP
space only to match dimensions: the Qwen and CLIP spaces are not aligned, so
code-involved directions measure an unaligned control. The original NetTAG
ExprLLM + TAGFormer model is not used.

Usage:
    python -m analogretriever.evaluation.evaluate_nettag --save_json results/nettag.json
"""
import argparse
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from analogretriever.evaluation.zero_shot import (content_fingerprint, encode_images_and_captions,
                                                  evaluate_baseline, load_clip_baseline, read_netlists,
                                                  spice_components)
from analogretriever.paths import repository_root

EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
INSTRUCTION = (
    "Given a text-attributed description of an analog circuit's component "
    "and connectivity, produce an embedding that captures the component's "
    "electrical role and local topological structure."
)
MAX_LENGTH = 4096
PROJECTION_SEED = 42


def device_attributes(spice_text):
    """One text attribute per device."""
    attributes = []
    for component in spice_components(spice_text):
        text = f"[Name] {component['name']} [Type] {component['type']}"
        if component['scope']:
            text += f" [Scope] subcircuit_{component['scope']}"
        terminals = component['terminals']
        connections = [f"{terminals[i] if i < len(terminals) else f'pin_{i}'}={net}"
                       for i, net in enumerate(component['nets'])]
        if connections:
            text += f" [Connections] {', '.join(connections)}"
        if component['parameters']:
            text += f" [Parameters] {' '.join(component['parameters'])}"
        attributes.append(text)
    return attributes


def last_token_pool(last_hidden_state, attention_mask):
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), lengths]


@torch.no_grad()
def embed_circuits(circuit_attributes, device, batch_size=8, tokenizer=None, model=None):
    """Encode each device separately, then mean-pool and normalize per circuit."""
    if not circuit_attributes or any(isinstance(attrs, str) or not attrs for attrs in circuit_attributes):
        raise ValueError("Each circuit needs a nonempty list of device attributes")
    if (tokenizer is None) != (model is None):
        raise ValueError("Supply both tokenizer and model, or neither")
    if model is None:
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
        model = AutoModel.from_pretrained(
            EMBEDDING_MODEL,
            torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32).to(device)
    model.eval()

    texts = [text for attributes in circuit_attributes for text in attributes]
    device_embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Qwen3 device embeddings"):
        chunk = [f"Instruct: {INSTRUCTION}\nQuery: {text}" for text in texts[start:start + batch_size]]
        encoded = tokenizer(chunk, padding=True, truncation=True, max_length=MAX_LENGTH,
                            return_tensors="pt").to(device)
        hidden = model(**encoded).last_hidden_state
        device_embeddings.append(F.normalize(last_token_pool(hidden, encoded["attention_mask"]).float(),
                                             p=2, dim=1).cpu())
    all_devices = torch.cat(device_embeddings)
    pooled, offset = [], 0
    for attributes in circuit_attributes:
        pooled.append(all_devices[offset:offset + len(attributes)].mean(dim=0))
        offset += len(attributes)
    return F.normalize(torch.stack(pooled), p=2, dim=1)


def project(embeddings, output_dim, seed=PROJECTION_SEED):
    """Fixed seeded random projection that only matches dimensionality."""
    if embeddings.shape[1] == output_dim:
        return F.normalize(embeddings, p=2, dim=1)
    generator = torch.Generator().manual_seed(seed)
    projection = torch.randn(embeddings.shape[1], output_dim, generator=generator,
                             dtype=embeddings.dtype).to(embeddings.device) / output_dim ** 0.5
    return F.normalize(embeddings @ projection, p=2, dim=1)


def cached_embeddings(cache_path, netlists, attributes, device):
    recipe = {"model": EMBEDDING_MODEL, "instruction": INSTRUCTION, "max_length": MAX_LENGTH,
              "pooling": "mean of normalized device embeddings, then normalized"}
    fingerprint = content_fingerprint({"netlists": netlists, "attributes": attributes}, recipe)
    cache_path = Path(cache_path)
    if cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu", weights_only=True)
        if (isinstance(cached, dict) and cached.get("input_fingerprint") == fingerprint
                and cached["embeddings"].shape[0] == len(netlists)):
            print(f"Using cached embeddings from {cache_path}")
            return cached["embeddings"]
    embeddings = embed_circuits(attributes, device)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"input_fingerprint": fingerprint, "embeddings": embeddings}, cache_path)
    return embeddings


def evaluate(test_csv, device, cache_path, save_json=None):
    root = repository_root()
    netlists = read_netlists(root, test_csv)
    attributes = [device_attributes(text) for text in netlists]
    circuit_embeddings = cached_embeddings(cache_path, netlists, attributes, device)

    clip_model, preprocess = load_clip_baseline(device)
    image_embeddings, caption_embeddings = encode_images_and_captions(clip_model, preprocess, root,
                                                                      test_csv, device)
    code_embeddings = project(circuit_embeddings.to(device), image_embeddings.shape[1])
    return evaluate_baseline("NetTAG", code_embeddings, image_embeddings, caption_embeddings, save_json)


def main():
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_csv", default=os.path.join(root, "test_data.csv"))
    parser.add_argument("--cache", default=os.path.join(root, ".cache", "nettag_embeddings.pt"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", default=None)
    args = parser.parse_args()
    evaluate(args.test_csv, args.device, args.cache, args.save_json)


if __name__ == "__main__":
    main()
