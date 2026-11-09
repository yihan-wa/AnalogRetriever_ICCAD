"""Train AnalogRetriever with the three-phase curriculum (paper Sec. 3.4).

Phase 1 (graph encoder warm-up): only the RGCN is trained, against detached
    CLIP embeddings, with the code-involved loss L_{I<->C} + L_{T<->C}.
Phase 2 (transition): CLIP is unfrozen and the six-direction loss L_tri is
    used; the optimizer is rebuilt with fresh Adam state and its own linear
    warm-up / cosine schedule. No intra-cluster negatives (alpha = 0).
Phase 3 (curriculum hard-negative mining): each batch is built around a seed
    circuit; an alpha_m fraction comes from the seed's functional cluster and
    alpha_m rises linearly from alpha_0 to alpha_max (Eq. 6).

Usage:
    python -m analogretriever.training.train_curriculum
"""
import argparse
import json
import os
import random

import numpy as np
import torch
from torch.utils.data import Sampler
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analogretriever.checkpoints import save_checkpoint
from analogretriever.encoders import clip_preprocess
from analogretriever.evaluation.protocol import SELECTION_DIRECTIONS, format_results, mean_recall_at_1
from analogretriever.paths import repository_root
from analogretriever.training.common import (
    CLIP_OPTIMIZER_MODES, TriModalModel, accumulation_group_size, build_optimizer,
    build_train_transform, code_alignment_loss, evaluate, evaluation_loader, load_split,
    new_logit_scale, optimizer_step, optimizer_updates_per_epoch, repository_file, set_seed,
    trimodal_loss, warmup_cosine_schedule,
)


class CurriculumBatchSampler(Sampler):
    """Batches of one seed circuit, ``floor(alpha * B)`` same-cluster circuits and
    circuits from the other clusters; a circuit appears at most once per batch."""

    def __init__(self, cluster_labels, batch_size, alpha=0.0):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if len(cluster_labels) == 0:
            raise ValueError("cluster_labels must not be empty")
        self.cluster_labels = np.array(cluster_labels, dtype=np.int64)
        self.batch_size = batch_size
        self.n = len(cluster_labels)
        self.set_alpha(alpha)
        self._clusters = {int(label): np.flatnonzero(self.cluster_labels == label)
                          for label in np.unique(self.cluster_labels)}
        sizes = [len(members) for members in self._clusters.values()]
        print(f"[Sampler] {self.n} circuits in {len(sizes)} clusters "
              f"(sizes {min(sizes)}-{max(sizes)})")

    def set_alpha(self, alpha):
        if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.alpha = float(alpha)

    def _sample_hard(self, seed_cluster, seed, count):
        pool = self._clusters[seed_cluster]
        pool = pool[pool != seed]
        return np.random.choice(pool, min(count, len(pool)), replace=False).tolist()

    def _sample_easy(self, seed_cluster, count):
        others = [members for label, members in self._clusters.items() if label != seed_cluster]
        if not others:
            return []
        pool = np.concatenate(others)
        return np.random.choice(pool, min(count, len(pool)), replace=False).tolist()

    def __iter__(self):
        seeds = np.random.permutation(self.n)
        for b in range(len(self)):
            size = self.batch_size
            num_hard = min(int(size * self.alpha), size - 1)
            seed = int(seeds[b])
            seed_cluster = int(self.cluster_labels[seed])
            hard = self._sample_hard(seed_cluster, seed, num_hard)
            batch = [seed] + hard + self._sample_easy(seed_cluster, size - 1 - len(hard))
            if len(batch) < size:
                unused = np.setdiff1d(np.arange(self.n), batch)
                batch.extend(np.random.choice(unused, size - len(batch), replace=False).tolist())
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n // self.batch_size


def curriculum_alpha(epoch, num_epochs, alpha_0, alpha_max):
    """alpha_m = min(alpha_max, alpha_0 + (m - 1) / (M - 1) * (alpha_max - alpha_0)) for m = 1..M (Eq. 6)."""
    if num_epochs <= 1:
        return alpha_max
    return min(alpha_max, alpha_0 + (epoch - 1) / (num_epochs - 1) * (alpha_max - alpha_0))


def load_cluster_labels(cluster_file, circuit_ids):
    with open(cluster_file, encoding="utf-8") as handle:
        clusters = json.load(handle)
    missing = [identity for identity in circuit_ids if identity not in clusters]
    if missing:
        raise ValueError(f"{len(missing)} training circuits have no cluster; rerun scripts/build_clusters.py")
    labels = [int(clusters[identity]) for identity in circuit_ids]
    if min(labels) < 0:
        raise ValueError("Cluster labels must be nonnegative; rerun scripts/build_clusters.py")
    return labels


def train(
    clip_lr=5e-5,
    graph_lr=5e-4,
    logit_scale_lr=1e-4,
    phase1_epochs=6,
    phase2_epochs=2,
    phase3_epochs=12,
    phase1_warmup_ratio=0.1,
    phase2_warmup_epochs=1,
    alpha_0=0.05,
    alpha_max=0.3,
    freeze_blocks=16,
    label_smoothing=0.1,
    batch_size=256,
    accum_steps=1,
    use_augmentation=True,
    clip_optimizer_mode="legacy_cuda_duplicate",
    seed=42,
    clusters="processed/train_clusters.json",
    output="checkpoints/analogretriever.pth",
):
    config = dict(locals())
    epochs = phase1_epochs + phase2_epochs + phase3_epochs
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = repository_root()
    train_csv = repository_file(root, "train_data.csv")
    test_csv = repository_file(root, "test_data.csv")
    cluster_file = os.path.join(root, clusters)
    if not os.path.isfile(cluster_file):
        raise FileNotFoundError(f"Missing {cluster_file}. Run scripts/build_clusters.py first.")

    preprocess = clip_preprocess()
    dataset = load_split(root, train_csv, build_train_transform(preprocess) if use_augmentation else preprocess)
    sampler = CurriculumBatchSampler(
        load_cluster_labels(cluster_file, dataset.df['id'].astype(str).tolist()), batch_size)
    loader = DataLoader(dataset, batch_sampler=sampler, num_workers=4)
    eval_loader = evaluation_loader(load_split(root, test_csv, preprocess))
    num_batches = len(loader)
    updates_per_epoch = optimizer_updates_per_epoch(num_batches, accum_steps)

    model = TriModalModel(conv="rgcn", device=device, freeze_blocks=freeze_blocks)
    logit_scale = new_logit_scale(device)

    def new_optimizer(total_steps, warmup_steps):
        optimizer = build_optimizer(model, logit_scale, clip_lr=clip_lr, graph_lr=graph_lr,
                                    logit_scale_lr=logit_scale_lr, device=device,
                                    clip_optimizer_mode=clip_optimizer_mode)
        return optimizer, warmup_cosine_schedule(optimizer, total_steps, warmup_steps)

    phase1_steps = updates_per_epoch * phase1_epochs
    optimizer, scheduler = new_optimizer(phase1_steps, int(phase1_steps * phase1_warmup_ratio))
    output_path = os.path.join(root, output)
    best_score, best_epoch = float("-inf"), None
    print(f"[AnalogRetriever] {epochs} epochs ({phase1_epochs}/{phase2_epochs}/{phase3_epochs} per phase), "
          f"batch {batch_size} x {accum_steps}, CLIP lr {clip_lr:g}, graph lr {graph_lr:g}, "
          f"alpha {alpha_0}->{alpha_max}, CLIP optimizer {clip_optimizer_mode}, seed {seed}")

    for epoch in range(1, epochs + 1):
        if epoch <= phase1_epochs:
            phase, alpha = 1, 0.0
        elif epoch <= phase1_epochs + phase2_epochs:
            phase, alpha = 2, 0.0
        else:
            phase = 3
            alpha = curriculum_alpha(epoch - phase1_epochs - phase2_epochs, phase3_epochs, alpha_0, alpha_max)
        if epoch == phase1_epochs + 1:
            joint_steps = updates_per_epoch * (epochs - phase1_epochs)
            optimizer, scheduler = new_optimizer(joint_steps, updates_per_epoch * phase2_warmup_epochs)
        sampler.set_alpha(alpha)

        model.train()
        total_loss, finite_steps = 0.0, 0
        progress = tqdm(enumerate(loader), total=num_batches, desc=f"Epoch {epoch}/{epochs} phase {phase}")
        for step, batch in progress:
            batch = batch.to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                v_i, v_t, v_c = model(batch)
                if phase == 1:
                    loss = code_alignment_loss(v_i.detach(), v_t.detach(), v_c, logit_scale, label_smoothing)
                else:
                    loss = trimodal_loss(v_i, v_t, v_c, logit_scale, label_smoothing)
                group_size = accumulation_group_size(step, num_batches, accum_steps)
                loss = loss / group_size
            loss.backward()
            if (step + 1) % accum_steps == 0 or step + 1 == num_batches:
                optimizer_step(model, optimizer, scheduler, logit_scale)
            step_loss = loss.item() * group_size
            if np.isfinite(step_loss):
                total_loss += step_loss
                finite_steps += 1
            progress.set_postfix(loss=f"{step_loss:.4f}", alpha=f"{alpha:.3f}")

        recall = evaluate(model, eval_loader, device)
        score = mean_recall_at_1(recall, SELECTION_DIRECTIONS)
        print(f"Epoch {epoch}: loss {total_loss / max(1, finite_steps):.4f}, "
              f"1/tau {logit_scale.exp().item():.2f}, alpha {alpha:.3f}, selection R@1 {score:.2f}\n"
              f"{format_results(recall)}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(model, logit_scale, output_path, model_type="AnalogRetriever",
                            training_config=config,
                            selection={"epoch": epoch, "directions": list(SELECTION_DIRECTIONS),
                                       "mean_r1": score, "recall": recall})
            print(f"Saved best checkpoint (epoch {epoch}) to {output_path}")

    print(f"Best selection R@1 {best_score:.2f} at epoch {best_epoch}: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip_lr", type=float, default=5e-5)
    parser.add_argument("--graph_lr", type=float, default=5e-4)
    parser.add_argument("--logit_scale_lr", type=float, default=1e-4)
    parser.add_argument("--phase1_epochs", type=int, default=6)
    parser.add_argument("--phase2_epochs", type=int, default=2)
    parser.add_argument("--phase3_epochs", type=int, default=12)
    parser.add_argument("--phase1_warmup_ratio", type=float, default=0.1)
    parser.add_argument("--phase2_warmup_epochs", type=int, default=1)
    parser.add_argument("--alpha_0", type=float, default=0.05)
    parser.add_argument("--alpha_max", type=float, default=0.3)
    parser.add_argument("--freeze_blocks", type=int, default=16,
                        help="Number of bottom ViT blocks kept frozen.")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--clip_optimizer_mode", default="legacy_cuda_duplicate", choices=CLIP_OPTIMIZER_MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clusters", default="processed/train_clusters.json")
    parser.add_argument("--output", default="checkpoints/analogretriever.pth")
    args = vars(parser.parse_args())
    args["use_augmentation"] = not args.pop("no_augmentation")
    train(**args)


if __name__ == "__main__":
    main()
