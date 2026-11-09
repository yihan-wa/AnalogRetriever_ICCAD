"""CLIP baseline of paper Table 2: raw SPICE text encoded by the CLIP text encoder.

Usage:
    python -m analogretriever.evaluation.evaluate_clip --save_json results/clip.json
"""
import argparse
import os

import torch

from analogretriever.evaluation.zero_shot import (encode_images_and_captions, encode_texts,
                                                  evaluate_baseline, load_clip_baseline, read_netlists)
from analogretriever.paths import repository_root

NETLIST_PREFIX_CHARS = 300


def evaluate(test_csv, device, save_json=None):
    root = repository_root()
    clip_model, preprocess = load_clip_baseline(device)
    image_embeddings, caption_embeddings = encode_images_and_captions(clip_model, preprocess, root,
                                                                      test_csv, device)
    netlists = [text[:NETLIST_PREFIX_CHARS] for text in read_netlists(root, test_csv)]
    return evaluate_baseline("CLIP", encode_texts(clip_model, netlists, device),
                             image_embeddings, caption_embeddings, save_json)


def main():
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_csv", default=os.path.join(root, "test_data.csv"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", default=None)
    args = parser.parse_args()
    evaluate(args.test_csv, args.device, args.save_json)


if __name__ == "__main__":
    main()
