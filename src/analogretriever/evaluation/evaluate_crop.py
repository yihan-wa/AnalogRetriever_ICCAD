"""CROP baseline of paper Table 2: each netlist is summarized by Qwen2.5-7B-Instruct
and the summary is encoded by the CLIP text encoder (a training-free
code-as-text baseline following CROP).

Summaries are cached in ``artifacts/crop_summaries.json``; a missing or
mismatching cache is regenerated, which needs a GPU.

Usage:
    python -m analogretriever.evaluation.evaluate_crop --save_json results/crop.json
"""
import argparse
import os

import torch

from analogretriever.evaluation.zero_shot import (NETLIST_PREFIX_CHARS, SUMMARY_MODEL, cached_summaries,
                                                  encode_images_and_captions, encode_texts,
                                                  evaluate_baseline, load_clip_baseline, read_netlists)
from analogretriever.paths import repository_root

PROMPT = """You are an expert analog circuit engineer. Given the following SPICE netlist, produce a one-sentence summary (under 50 words) describing the circuit type, key components, and primary function.

SPICE Netlist:
{netlist}

Summary:"""
MAX_NEW_TOKENS = 80


def evaluate(test_csv, device, cache_path, save_json=None):
    root = repository_root()
    netlists = read_netlists(root, test_csv)
    recipe = {"prompt": PROMPT, "model": SUMMARY_MODEL, "decoding": "greedy",
              "max_new_tokens": MAX_NEW_TOKENS, "netlist_prefix_chars": NETLIST_PREFIX_CHARS}
    prompts = [PROMPT.format(netlist=text[:NETLIST_PREFIX_CHARS]) for text in netlists]
    summaries = cached_summaries(cache_path, netlists, prompts, recipe, MAX_NEW_TOKENS, device)

    clip_model, preprocess = load_clip_baseline(device)
    image_embeddings, caption_embeddings = encode_images_and_captions(clip_model, preprocess, root,
                                                                      test_csv, device)
    return evaluate_baseline("CROP", encode_texts(clip_model, summaries, device),
                             image_embeddings, caption_embeddings, save_json)


def main():
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_csv", default=os.path.join(root, "test_data.csv"))
    parser.add_argument("--summaries", default=os.path.join(root, "artifacts", "crop_summaries.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", default=None)
    args = parser.parse_args()
    evaluate(args.test_csv, args.device, args.summaries, args.save_json)


if __name__ == "__main__":
    main()
