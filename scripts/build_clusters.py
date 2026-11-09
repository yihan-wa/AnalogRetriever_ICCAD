"""Functional clusters for curriculum hard-negative mining (paper Sec. 3.4).

Every training netlist is embedded with Qwen3-Embedding and the normalized
embeddings are grouped by K-means (K = 30). The result maps circuit IDs to
cluster labels and is read by ``analogretriever.training.train_curriculum``.

Usage:
    python scripts/build_clusters.py
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

from analogretriever.paths import repository_root

INSTRUCTION = ("Given a SPICE netlist describing an analog circuit, "
               "retrieve netlists that implement functionally similar circuits.")


def last_token_pool(last_hidden_state, attention_mask):
    if attention_mask[:, -1].sum() == attention_mask.shape[0]:
        return last_hidden_state[:, -1]
    lengths = attention_mask.sum(dim=1) - 1
    return last_hidden_state[torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device), lengths]


@torch.no_grad()
def embed_netlists(texts, model_name, device, batch_size, max_length):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32).to(device).eval()
    embeddings = []
    for start in tqdm(range(0, len(texts), batch_size), desc="Embedding netlists"):
        chunk = [f"Instruct: {INSTRUCTION}\nQuery: {text}" for text in texts[start:start + batch_size]]
        encoded = tokenizer(chunk, padding=True, truncation=True, max_length=max_length,
                            return_tensors="pt").to(device)
        hidden = model(**encoded).last_hidden_state
        embeddings.append(F.normalize(last_token_pool(hidden, encoded["attention_mask"]), p=2, dim=1)
                          .float().cpu().numpy())
    return np.concatenate(embeddings)


def build_clusters(model_name, k, batch_size, max_length, output, device):
    root = Path(repository_root())
    frame = pd.read_csv(root / "train_data.csv")
    texts = [(root / path).read_text(encoding="utf-8", errors="replace").strip() for path in frame["spice"]]
    features = embed_netlists(texts, model_name, device, batch_size, max_length)
    labels = KMeans(n_clusters=k, random_state=42, n_init="auto").fit_predict(features)
    sizes = np.bincount(labels, minlength=k)
    print(f"{len(labels)} training circuits in {k} clusters (sizes {sizes.min()}-{sizes.max()})")
    output = root / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({str(i): int(label) for i, label in zip(frame["id"], labels)}, indent=1) + "\n")
    print(f"Saved cluster assignments to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--k", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--output", default="processed/train_clusters.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    build_clusters(args.model, args.k, args.batch_size, args.max_length, args.output, args.device)


if __name__ == "__main__":
    main()
