"""Model, loss, optimizer and evaluation components shared by the TI, TIC and
AnalogRetriever training scripts (paper Sec. 3.3, 3.4 and 4.2)."""
import random
from pathlib import Path

import clip
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torchvision import transforms

from analogretriever.dataset import TriModalCircuitDataset
from analogretriever.encoders import ImageEncoder, NetlistEncoder, TextEncoder, load_clip
from analogretriever.evaluation.protocol import fixed_protocol_recall

# Registration of the shared CLIP parameters with AdamW. The image and text
# wrappers hold the same CLIP model, so "legacy_cuda_duplicate" registers each
# trainable CLIP parameter twice (once per wrapper). On CUDA, AdamW then
# applies two updates per optimizer step to these parameters; this is the
# configuration used for the released checkpoints and requires a CUDA device.
# "standard" registers each parameter once.
CLIP_OPTIMIZER_MODES = ("legacy_cuda_duplicate", "standard")
WEIGHT_DECAY = 1e-4
LOGIT_SCALE_INIT = 1 / 0.07
MAX_LOGIT_SCALE = 100.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_train_transform(clip_preprocess):
    """Mild augmentation for schematic line drawings, followed by CLIP normalization."""
    normalize = next(t for t in clip_preprocess.transforms if isinstance(t, transforms.Normalize))
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])


def freeze_vision_blocks(visual, num_blocks):
    """Freeze the patch embedding, positional embeddings and the bottom ``num_blocks`` ViT blocks."""
    frozen = [visual.conv1.weight, visual.class_embedding, visual.positional_embedding,
              *visual.ln_pre.parameters()]
    for block in visual.transformer.resblocks[:num_blocks]:
        frozen.extend(block.parameters())
    for parameter in frozen:
        parameter.requires_grad = False
    total_blocks = len(visual.transformer.resblocks)
    trainable = sum(p.numel() for p in visual.parameters() if p.requires_grad)
    print(f"[Freeze] ViT: {min(num_blocks, total_blocks)}/{total_blocks} blocks frozen, "
          f"{trainable / 1e6:.1f}M trainable parameters")


class BiModalModel(nn.Module):
    """CLIP image and text encoders f_I and f_T (TI ablation)."""

    def __init__(self, device="cuda", freeze_blocks=16, clip_state_dict=None):
        super().__init__()
        clip_model, _ = load_clip(device, clip_state_dict)
        self.image_encoder = ImageEncoder(clip_model)
        self.text_encoder = TextEncoder(clip_model)
        if freeze_blocks > 0:
            freeze_vision_blocks(clip_model.visual, freeze_blocks)

    def forward(self, batch):
        return (self.image_encoder(batch.image_tensor.squeeze(1)),
                self.text_encoder(batch.text_tokens.squeeze(1)))


class TriModalModel(BiModalModel):
    """CLIP encoders plus the netlist graph encoder f_C (TIC and AnalogRetriever)."""

    def __init__(self, conv="rgcn", device="cuda", freeze_blocks=16, clip_state_dict=None):
        super().__init__(device, freeze_blocks, clip_state_dict)
        self.netlist_encoder = NetlistEncoder(conv=conv).to(device)

    def forward(self, batch):
        v_i, v_t = super().forward(batch)
        v_c = self.netlist_encoder(batch.x_type, batch.x_continuous, batch.edge_index,
                                   batch.batch, batch.edge_attr)
        return v_i, v_t, v_c


def new_logit_scale(device):
    """Learnable logit scale 1/tau, initialised to 1/0.07."""
    return nn.Parameter(torch.ones([], device=device) * np.log(LOGIT_SCALE_INIT))


def contrastive_loss(a, b, logit_scale, label_smoothing):
    """L_{a->b} + L_{b->a}: symmetric InfoNCE over a batch of aligned pairs (Eq. 4)."""
    logits = (a @ b.t()) * torch.clamp(logit_scale.exp(), max=MAX_LOGIT_SCALE)
    labels = torch.arange(a.shape[0], device=a.device)
    return (F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
            + F.cross_entropy(logits.t(), labels, label_smoothing=label_smoothing))


def code_alignment_loss(v_i, v_t, v_c, logit_scale, label_smoothing):
    """Phase-1 loss L_{I<->C} + L_{T<->C}; pass detached CLIP embeddings."""
    return (contrastive_loss(v_i, v_c, logit_scale, label_smoothing)
            + contrastive_loss(v_t, v_c, logit_scale, label_smoothing))


def trimodal_loss(v_i, v_t, v_c, logit_scale, label_smoothing):
    """Six-direction tri-modal loss L_tri over (I, T), (I, C) and (T, C) (Eq. 5)."""
    return (contrastive_loss(v_i, v_t, logit_scale, label_smoothing)
            + contrastive_loss(v_i, v_c, logit_scale, label_smoothing)
            + contrastive_loss(v_t, v_c, logit_scale, label_smoothing))


def clip_parameters(model, mode, device):
    if mode not in CLIP_OPTIMIZER_MODES:
        raise ValueError(f"Unknown clip_optimizer_mode {mode!r}; choose from {CLIP_OPTIMIZER_MODES}")
    parameters = [p for p in (list(model.image_encoder.parameters())
                              + list(model.text_encoder.parameters())) if p.requires_grad]
    if mode == "legacy_cuda_duplicate":
        if torch.device(device).type != "cuda":
            raise ValueError("clip_optimizer_mode='legacy_cuda_duplicate' requires CUDA; "
                             "use 'standard' on other devices.")
        return parameters
    return list({id(p): p for p in parameters}.values())


def build_optimizer(model, logit_scale, *, clip_lr, logit_scale_lr, clip_optimizer_mode,
                    device, graph_lr=None):
    groups = [{"params": clip_parameters(model, clip_optimizer_mode, device), "lr": clip_lr},
              {"params": [logit_scale], "lr": logit_scale_lr}]
    if graph_lr is not None:
        groups.append({"params": list(model.netlist_encoder.parameters()), "lr": graph_lr})
    return torch.optim.AdamW(groups, weight_decay=WEIGHT_DECAY)


def warmup_cosine_schedule(optimizer, total_steps, warmup_steps):
    """Linear warm-up followed by cosine decay to 1% of the base learning rate."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.01, 0.5 * (1.0 + np.cos(np.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def optimizer_updates_per_epoch(num_batches, accum_steps):
    if accum_steps < 1:
        raise ValueError("accum_steps must be positive")
    if num_batches < 1:
        raise ValueError("Training has no batches; check the data and batch size")
    return (num_batches + accum_steps - 1) // accum_steps


def accumulation_group_size(step, num_batches, accum_steps):
    """Number of micro-batches in the update containing ``step``, including a shorter tail."""
    if not 0 <= step < num_batches:
        raise ValueError("step is outside this epoch")
    return min(accum_steps, num_batches - (step // accum_steps) * accum_steps)


def optimizer_step(model, optimizer, scheduler, logit_scale):
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()
    with torch.no_grad():
        logit_scale.clamp_(0.0, np.log(MAX_LOGIT_SCALE))


def load_split(root, csv_file, image_preprocess):
    return TriModalCircuitDataset(root=root, csv_file=csv_file, tokenizer=clip.tokenize,
                                  image_preprocess=image_preprocess)


def evaluation_loader(dataset, batch_size=64, num_workers=4):
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                      generator=torch.Generator().manual_seed(0))


@torch.no_grad()
def encode_dataset(model, loader, device):
    """Row-aligned float32 embeddings ``{"I": ..., "T": ..., "C": ...}`` for every item."""
    was_training = model.training
    model.eval()
    outputs = None
    for batch in loader:
        embeddings = model(batch.to(device))
        if outputs is None:
            outputs = [[] for _ in embeddings]
        for collected, value in zip(outputs, embeddings):
            collected.append(value.float())
    model.train(was_training)
    return dict(zip("ITC", (torch.cat(values) for values in outputs)))


def evaluate(model, loader, device):
    return fixed_protocol_recall(encode_dataset(model, loader, device))


def repository_file(root, name):
    path = Path(root) / name
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run scripts/prepare_dataset.py first.")
    return str(path)
