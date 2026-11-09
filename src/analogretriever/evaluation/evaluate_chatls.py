"""ChatLS baseline of paper Table 2: an LLM-based structured netlist representation.

Device counts, net counts and named supply/ground nets are extracted from the
netlist and given to Qwen2.5-7B-Instruct together with the SPICE text; the
resulting summary is encoded by the CLIP text encoder. Unlike the original
ChatLS, no learned circuit-graph model is used.

Summaries are cached in ``artifacts/chatls_summaries.json``; a missing or
mismatching cache is regenerated, which needs a GPU.

Usage:
    python -m analogretriever.evaluation.evaluate_chatls --save_json results/chatls.json
"""
import argparse
import os
from collections import Counter

import torch

from analogretriever.evaluation.zero_shot import (NETLIST_PREFIX_CHARS, SUMMARY_MODEL, cached_summaries,
                                                  encode_images_and_captions, encode_texts,
                                                  evaluate_baseline, load_clip_baseline, read_netlists,
                                                  spice_components)
from analogretriever.paths import repository_root

PROMPT = """You are an expert analog circuit engineer with deep understanding of circuit topology and function. Given the following SPICE netlist and its structural analysis, produce a one-sentence summary (under 60 words) describing the circuit type, topology, key components, and primary function.

Structural Analysis:
{analysis}

SPICE Netlist:
{netlist}

Summary:"""
MAX_NEW_TOKENS = 100


def structural_analysis(spice_text):
    """Device counts, net count and named supply/ground nets as one sentence group."""
    device_counts = Counter()
    nets = set()
    for component in spice_components(spice_text):
        device_counts[component['type']] += 1
        for net in component['nets']:
            nets.add((0 if net == '0' else component['scope'], net))

    parts = [f"Circuit has {sum(device_counts.values())} devices and {len(nets)} nets."]
    if device_counts:
        listing = ", ".join(f"{count} {name}" for name, count in
                            sorted(device_counts.items(), key=lambda item: -item[1]))
        parts.append(f"Components: {listing}.")
    supply = sorted({net for _, net in nets if 'VDD' in net or 'VCC' in net})
    ground = sorted({net for _, net in nets if 'VSS' in net or 'GND' in net or net == '0'})
    if supply:
        parts.append(f"Power: {', '.join(supply[:3])}.")
    if ground:
        parts.append(f"Ground: {', '.join(ground[:3])}.")
    return " ".join(parts)


def evaluate(test_csv, device, cache_path, save_json=None):
    root = repository_root()
    netlists = read_netlists(root, test_csv)
    analyses = [structural_analysis(text) for text in netlists]
    recipe = {"prompt": PROMPT, "model": SUMMARY_MODEL, "decoding": "greedy",
              "max_new_tokens": MAX_NEW_TOKENS, "netlist_prefix_chars": NETLIST_PREFIX_CHARS,
              "analyses": analyses}
    prompts = [PROMPT.format(analysis=analysis, netlist=text[:NETLIST_PREFIX_CHARS])
               for analysis, text in zip(analyses, netlists)]
    summaries = cached_summaries(cache_path, netlists, prompts, recipe, MAX_NEW_TOKENS, device)

    clip_model, preprocess = load_clip_baseline(device)
    image_embeddings, caption_embeddings = encode_images_and_captions(clip_model, preprocess, root,
                                                                      test_csv, device)
    return evaluate_baseline("ChatLS", encode_texts(clip_model, summaries, device),
                             image_embeddings, caption_embeddings, save_json)


def main():
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test_csv", default=os.path.join(root, "test_data.csv"))
    parser.add_argument("--summaries", default=os.path.join(root, "artifacts", "chatls_summaries.json"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_json", default=None)
    args = parser.parse_args()
    evaluate(args.test_csv, args.device, args.summaries, args.save_json)


if __name__ == "__main__":
    main()
