"""Shared pieces of the zero-shot baselines in paper Table 2 (CLIP, CROP, ChatLS, NetTAG).

All four baselines encode schematics and descriptions with the off-the-shelf
CLIP ViT-L/14 and differ only in how the netlist ("code") modality is
represented, so they share identical Text->Image and Image->Text recall.
CROP, ChatLS and NetTAG were designed for other circuit domains; the
functions here implement their netlist representations for analog SPICE as
described in the paper and README.
"""
import hashlib
import json
from pathlib import Path

import clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from analogretriever.dataset import (DEVICE_REGISTRY, DEVICE_TERMINALS, MODEL_POLARITIES,
                                     bjt_terminal_count, is_net_token, netlist_records)
from analogretriever.encoders import CLIP_MODEL_NAME
from analogretriever.evaluation.protocol import fixed_protocol_recall, format_results, save_results

SUMMARY_MODEL = "Qwen/Qwen2.5-7B-Instruct"
NETLIST_PREFIX_CHARS = 1500

BASELINES = {
    "CLIP": "Raw SPICE text (first 300 characters) encoded by the CLIP text encoder.",
    "CROP": f"SPICE netlist summarized by {SUMMARY_MODEL} (greedy decoding), summary encoded by CLIP.",
    "ChatLS": f"Device/net statistics plus SPICE summarized by {SUMMARY_MODEL} (greedy decoding), "
              "summary encoded by CLIP.",
    "NetTAG": "Per-device text attributes encoded by Qwen3-Embedding-0.6B, mean-pooled, and mapped "
              "to the CLIP dimension by a fixed seeded random projection (no alignment training).",
}


class ImageCaptionDataset(Dataset):
    def __init__(self, root, frame, preprocess):
        self.root, self.frame, self.preprocess = Path(root), frame, preprocess

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(self.root / row['image']) as image:
            pixels = self.preprocess(image.convert("RGB"))
        return pixels, clip.tokenize([str(row['caption'])], truncate=True)[0]


def read_netlists(root, csv_file):
    frame = pd.read_csv(csv_file)
    return [(Path(root) / path).read_text(encoding="utf-8").strip() for path in frame['spice']]


@torch.no_grad()
def encode_texts(clip_model, texts, device, batch_size=64):
    features = []
    for start in tqdm(range(0, len(texts), batch_size), desc="CLIP text"):
        tokens = clip.tokenize(texts[start:start + batch_size], truncate=True).to(device)
        value = clip_model.encode_text(tokens)
        features.append((value / value.norm(dim=-1, keepdim=True)).float())
    return torch.cat(features)


@torch.no_grad()
def encode_images_and_captions(clip_model, preprocess, root, csv_file, device, batch_size=64):
    loader = DataLoader(ImageCaptionDataset(root, pd.read_csv(csv_file), preprocess),
                        batch_size=batch_size, shuffle=False, num_workers=4)
    images, captions = [], []
    for pixels, tokens in tqdm(loader, desc="CLIP image + caption"):
        v_i = clip_model.encode_image(pixels.to(device))
        v_t = clip_model.encode_text(tokens.to(device))
        images.append((v_i / v_i.norm(dim=-1, keepdim=True)).float())
        captions.append((v_t / v_t.norm(dim=-1, keepdim=True)).float())
    return torch.cat(images), torch.cat(captions)


def load_clip_baseline(device):
    clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    return clip_model.eval(), preprocess


def evaluate_baseline(name, code_embeddings, image_embeddings, caption_embeddings, save_json=None):
    results = fixed_protocol_recall({"I": image_embeddings, "T": caption_embeddings,
                                     "C": code_embeddings.to(image_embeddings.device)})
    print(f"{name} (zero-shot):\n{format_results(results)}")
    if save_json:
        save_results(results, save_json, {"model": name, "implementation": BASELINES[name]})
        print(f"Saved results to {save_json}")
    return results


def content_fingerprint(inputs, recipe):
    content = json.dumps({"inputs": inputs, "recipe": recipe}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_summaries(path, fingerprint, count):
    """Cached summaries, or None if the file is missing or was built from other inputs."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("input_fingerprint") != fingerprint:
        return None
    summaries = data.get("summaries")
    if (not isinstance(summaries, list) or len(summaries) != count
            or not all(isinstance(s, str) and s.strip() for s in summaries)):
        return None
    return summaries


def save_summaries(path, fingerprint, summaries, recipe):
    if not all(isinstance(s, str) and s.strip() for s in summaries):
        raise ValueError("Refusing to cache empty generated summaries")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"input_fingerprint": fingerprint, "recipe": recipe, "summaries": summaries}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


@torch.no_grad()
def generate_summaries(prompts, max_new_tokens, device, model_name=SUMMARY_MODEL):
    """Greedy chat completions, one prompt at a time."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map=device,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32).eval()
    summaries = []
    for prompt in tqdm(prompts, desc=f"{model_name} summaries"):
        text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                             tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([text], return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        summaries.append(tokenizer.decode(output[0][inputs.input_ids.shape[1]:],
                                          skip_special_tokens=True).strip())
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return summaries


def cached_summaries(cache_path, netlists, prompts, recipe, max_new_tokens, device):
    fingerprint = content_fingerprint(netlists, recipe)
    summaries = load_summaries(cache_path, fingerprint, len(netlists))
    if summaries is None:
        print(f"No matching summary cache at {cache_path}; generating {len(prompts)} summaries")
        summaries = generate_summaries(prompts, max_new_tokens, device)
        save_summaries(cache_path, fingerprint, summaries, recipe)
    else:
        print(f"Using cached summaries from {cache_path}")
    return summaries


def spice_components(spice_text):
    """Device records (name, type, nets, terminal names, trailing parameters, scope) of a netlist.

    Uses the same instance grammar as the graph parser; mutual-inductance (K)
    lines are kept as devices without terminals.
    """
    scope_ids = {}
    for line, models in netlist_records(spice_text.splitlines()):
        if line.startswith('.') or line.lower() == 'spice':
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        name = parts[0].upper()
        prefix = name[0]
        if prefix in DEVICE_REGISTRY:
            device_type, terminal_count = DEVICE_REGISTRY[prefix]
            terminal_names = DEVICE_TERMINALS[prefix]
        elif prefix == 'X':
            device_type, terminal_count = 'SUBCIRCUIT', len(parts) - 2
            terminal_names = ('subckt_terminal',) * terminal_count
        elif prefix == 'K':
            device_type, terminal_count, terminal_names = 'MUTUAL_INDUCTANCE', 0, ()
        else:
            continue
        if prefix == 'Q':
            terminal_count = bjt_terminal_count(parts, models)
        nets = []
        for token in parts[1:]:
            if len(nets) == terminal_count or not is_net_token(token):
                break
            nets.append(token.upper())
        parameters = parts[1 + len(nets):]
        if prefix in MODEL_POLARITIES and parameters:
            model_name = parameters[0].upper().strip('{}')
            declared_type = models.get(model_name, model_name)
            if declared_type in MODEL_POLARITIES[prefix]:
                device_type = declared_type
        elif prefix == 'B' and parameters:
            device_type = 'I_SOURCE' if parameters[0].upper().startswith('I') else 'V_SOURCE'
        scope_ids.setdefault(id(models.maps[-1]), 0)
        scope = scope_ids.setdefault(id(models.maps[0]), len(scope_ids))
        yield {'name': name, 'type': device_type, 'nets': nets,
               'terminals': terminal_names, 'parameters': parameters, 'scope': scope}
