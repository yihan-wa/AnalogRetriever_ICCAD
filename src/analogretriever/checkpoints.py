"""Checkpoint files: a PyTorch state dict plus a ``<checkpoint>.metadata.json`` sidecar."""
import json
import platform
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch

from analogretriever.dataset import (CONTINUOUS_FEATURES, GRAPH_SCHEMA_VERSION, NODE_TYPES,
                                     RELATION_TYPES, file_sha256)

# Model names follow the rows of paper Table 2.
MODEL_TYPES = ("TI", "TIC (GCN)", "TIC (RGCN)", "AnalogRetriever")
FORMAT_VERSION = 1


def metadata_path(checkpoint):
    return Path(str(checkpoint) + ".metadata.json")


def _source_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions():
    versions = {}
    for name in ("torch", "torch-geometric", "torchvision", "clip", "numpy"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def save_checkpoint(model, logit_scale, path, *, model_type, training_config, selection):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"model_type must be one of {MODEL_TYPES}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    state["logit_scale"] = logit_scale.detach()
    torch.save(state, path)
    metadata = {
        "format_version": FORMAT_VERSION,
        "model_type": model_type,
        "training_config": training_config,
        "selection": selection,
        "source_commit": _source_commit(),
        "python": platform.python_version(),
        "packages": _package_versions(),
        "cuda": torch.version.cuda,
        "checkpoint_sha256": file_sha256(path),
    }
    if model_type != "TI":
        metadata["graph"] = {"schema_version": GRAPH_SCHEMA_VERSION, "node_types": list(NODE_TYPES),
                             "relation_types": list(RELATION_TYPES),
                             "continuous_features": list(CONTINUOUS_FEATURES)}
    metadata_path(path).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


def infer_model_type(state_dict):
    if "netlist_encoder.type_emb.weight" not in state_dict:
        return "TI"
    return "TIC (RGCN)" if "netlist_encoder.conv1.weight" in state_dict else "TIC (GCN)"


def _check_graph_compatibility(state_dict, metadata):
    num_node_types = state_dict["netlist_encoder.type_emb.weight"].shape[0]
    if num_node_types != len(NODE_TYPES):
        raise ValueError(f"Checkpoint has {num_node_types} node types; this code defines {len(NODE_TYPES)}.")
    relation_weights = state_dict.get("netlist_encoder.conv1.weight")
    if relation_weights is not None and relation_weights.shape[0] != len(RELATION_TYPES):
        raise ValueError(f"Checkpoint has {relation_weights.shape[0]} relation types; "
                         f"this code defines {len(RELATION_TYPES)}.")
    graph = metadata.get("graph") if metadata else None
    if graph and (graph.get("schema_version") != GRAPH_SCHEMA_VERSION
                  or graph.get("node_types") != list(NODE_TYPES)
                  or graph.get("relation_types") != list(RELATION_TYPES)):
        raise ValueError("Checkpoint graph schema differs from this code's netlist graph definition.")


def load_checkpoint(path, device="cpu", verify_hash=True):
    """Rebuild the model stored in ``path``; the CLIP weights come from the checkpoint itself.

    Returns ``(model, logit_scale, metadata)``. The sidecar is optional; when
    present its hash and graph schema are verified.
    """
    from analogretriever.training.common import BiModalModel, TriModalModel

    path = Path(path)
    sidecar = metadata_path(path)
    metadata = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else None
    if metadata and verify_hash and metadata.get("checkpoint_sha256") not in (None, file_sha256(path)):
        raise ValueError(f"{path} does not match the checksum recorded in {sidecar}")

    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    logit_scale = state_dict.pop("logit_scale", None)
    model_type = metadata["model_type"] if metadata else infer_model_type(state_dict)
    prefix = "image_encoder.model."
    clip_state = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    if model_type == "TI":
        model = BiModalModel(device=device, freeze_blocks=0, clip_state_dict=clip_state)
    else:
        _check_graph_compatibility(state_dict, metadata)
        conv = "rgcn" if "netlist_encoder.conv1.weight" in state_dict else "gcn"
        model = TriModalModel(conv=conv, device=device, freeze_blocks=0, clip_state_dict=clip_state)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval(), logit_scale, metadata or {"model_type": model_type}
