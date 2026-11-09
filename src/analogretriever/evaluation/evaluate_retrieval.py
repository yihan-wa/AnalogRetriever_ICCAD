"""Evaluate a trained TI, TIC or AnalogRetriever checkpoint on the held-out split.

Usage:
    python -m analogretriever.evaluation.evaluate_retrieval \
        --checkpoint checkpoints/analogretriever.pth --save_json results/analogretriever.json
"""
import argparse
import os

import torch

from analogretriever.checkpoints import load_checkpoint
from analogretriever.encoders import clip_preprocess
from analogretriever.evaluation.protocol import fixed_protocol_recall, format_results, save_results
from analogretriever.paths import repository_root
from analogretriever.training.common import encode_dataset, evaluation_loader, load_split


def evaluate_checkpoint(checkpoint, test_csv, device="cuda", batch_size=64, num_workers=4):
    model, logit_scale, metadata = load_checkpoint(checkpoint, device=device)
    print(f"Loaded {metadata['model_type']} from {checkpoint}")
    if logit_scale is not None:
        print(f"Learned temperature tau = {1.0 / logit_scale.exp().item():.4f}")
    dataset = load_split(repository_root(), test_csv, clip_preprocess())
    embeddings = encode_dataset(model, evaluation_loader(dataset, batch_size, num_workers), device)
    results = fixed_protocol_recall(embeddings)
    print(format_results(results))
    return results, metadata


def main():
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", default=os.path.join(root, "checkpoints", "analogretriever.pth"))
    parser.add_argument("--test_csv", default=os.path.join(root, "test_data.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_json", default=None)
    args = parser.parse_args()
    results, metadata = evaluate_checkpoint(args.checkpoint, args.test_csv, args.device,
                                            args.batch_size, args.num_workers)
    if args.save_json:
        save_results(results, args.save_json, {
            "model": metadata["model_type"], "checkpoint": os.path.basename(args.checkpoint),
            "checkpoint_sha256": metadata.get("checkpoint_sha256"), "device": args.device,
        })
        print(f"Saved results to {args.save_json}")


if __name__ == "__main__":
    main()
