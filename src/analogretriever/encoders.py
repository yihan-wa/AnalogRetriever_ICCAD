"""Modality encoders (paper Sec. 3.2): CLIP ViT-L/14 for schematics and
descriptions, and a port-aware relational graph network for netlists."""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GraphNorm, RGCNConv, global_add_pool
from torch_geometric.utils import softmax

from analogretriever.dataset import NODE_TYPES, NUM_CONTINUOUS_FEATURES, NUM_RELATIONS

CLIP_MODEL_NAME = "ViT-L/14"
CLIP_INPUT_RESOLUTION = 224


class ImageEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, images):
        return F.normalize(self.model.encode_image(images), p=2, dim=1)


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.model = clip_model

    def forward(self, text_tokens):
        return F.normalize(self.model.encode_text(text_tokens), p=2, dim=1)


class AttentionPooling(nn.Module):
    """Graph readout g = sum_v alpha_v h_v with alpha = softmax over each graph (Eq. 3).

    The attention logits are scaled dot products between a learned query and
    projected node states, i.e. a learned linear scoring of h_v.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.super_node = nn.Parameter(torch.randn(1, hidden_dim))
        self.scale = hidden_dim ** -0.5

    def forward(self, x, batch):
        query = self.q_proj(self.super_node)
        scores = (self.k_proj(x) @ query.t()).squeeze(-1) * self.scale
        weights = softmax(scores, batch)
        return global_add_pool(x * weights.unsqueeze(-1), batch)


class NetlistEncoder(nn.Module):
    """Netlist graph encoder f_C.

    Node features fuse a node-type embedding with log(1 + x) continuous
    parameters (Eq. 1). Two message-passing layers with GraphNorm, GELU and
    residual connections (Eq. 2) are followed by attention pooling (Eq. 3) and
    a two-layer MLP projection into the CLIP embedding space.

    ``conv="rgcn"`` uses one weight matrix per relation type (AnalogRetriever,
    TIC (RGCN)); ``conv="gcn"`` ignores relation types (TIC (GCN) ablation).
    """

    def __init__(self, conv="rgcn", num_node_types=len(NODE_TYPES),
                 num_relations=NUM_RELATIONS, num_continuous_features=NUM_CONTINUOUS_FEATURES,
                 hidden_dim=512, out_dim=768):
        super().__init__()
        if conv not in ("rgcn", "gcn"):
            raise ValueError(f"conv must be 'rgcn' or 'gcn', got {conv!r}")
        self.conv_type = conv
        self.type_emb = nn.Embedding(num_node_types, hidden_dim)
        self.continuous_proj = nn.Linear(num_continuous_features, hidden_dim)
        self.fuse = nn.Linear(hidden_dim * 2, hidden_dim)
        if conv == "rgcn":
            self.conv1 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
            self.conv2 = RGCNConv(hidden_dim, hidden_dim, num_relations=num_relations)
        else:
            self.conv1 = GCNConv(hidden_dim, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.gn1 = GraphNorm(hidden_dim)
        self.gn2 = GraphNorm(hidden_dim)
        self.readout = AttentionPooling(hidden_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, out_dim),
        )

    def _conv(self, layer, h, edge_index, edge_type):
        if self.conv_type == "rgcn":
            return layer(h, edge_index, edge_type=edge_type)
        return layer(h, edge_index)

    def forward(self, x_type, x_continuous, edge_index, batch, edge_type):
        continuous = torch.log1p(torch.clamp(x_continuous, min=0.0))
        h = self.fuse(torch.cat([self.type_emb(x_type), self.continuous_proj(continuous)], dim=-1))
        for conv, norm in ((self.conv1, self.gn1), (self.conv2, self.gn2)):
            h = F.gelu(norm(self._conv(conv, h, edge_index, edge_type), batch)) + h
        return F.normalize(self.projection_head(self.readout(h, batch)), p=2, dim=1)


def load_clip(device="cpu", state_dict=None):
    """Return a float32 CLIP ViT-L/14 and its evaluation preprocessing.

    With ``state_dict`` the model is rebuilt from those weights (for example
    the CLIP weights stored in an AnalogRetriever checkpoint) instead of
    downloading the OpenAI checkpoint.
    """
    if state_dict is None:
        clip_model, preprocess = clip.load(CLIP_MODEL_NAME, device=device, jit=False)
    else:
        clip_model = clip.model.build_model(dict(state_dict))
        preprocess = clip.clip._transform(clip_model.visual.input_resolution)
    return clip_model.float().to(device), preprocess


def clip_preprocess():
    """CLIP's evaluation transform (resize, center crop, normalize) without loading weights."""
    return clip.clip._transform(CLIP_INPUT_RESOLUTION)
