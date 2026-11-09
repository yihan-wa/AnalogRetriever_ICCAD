"""Train the TI (bi-modal) ablation of paper Table 2.

TI fine-tunes CLIP on image-text pairs only, with the same backbone, frozen
ViT blocks and 20-epoch budget as the tri-modal models. The checkpoint is
selected by held-out Text->Image R@1.

Usage:
    python -m analogretriever.training.train_bimodal
"""
import argparse
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from analogretriever.checkpoints import save_checkpoint
from analogretriever.encoders import clip_preprocess
from analogretriever.evaluation.protocol import format_results
from analogretriever.paths import repository_root
from analogretriever.training.common import (
    CLIP_OPTIMIZER_MODES, BiModalModel, accumulation_group_size, build_optimizer,
    build_train_transform, contrastive_loss, evaluate, evaluation_loader, load_split,
    new_logit_scale, optimizer_step, optimizer_updates_per_epoch, repository_file, set_seed,
    warmup_cosine_schedule,
)


def train(
    clip_lr=5e-5,
    logit_scale_lr=5e-5,
    epochs=20,
    warmup_ratio=0.1,
    freeze_blocks=16,
    label_smoothing=0.1,
    batch_size=256,
    accum_steps=1,
    use_augmentation=True,
    clip_optimizer_mode="legacy_cuda_duplicate",
    seed=42,
    output="checkpoints/ti.pth",
):
    config = dict(locals())
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = repository_root()
    train_csv = repository_file(root, "train_data.csv")
    test_csv = repository_file(root, "test_data.csv")

    preprocess = clip_preprocess()
    dataset = load_split(root, train_csv, build_train_transform(preprocess) if use_augmentation else preprocess)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)
    eval_loader = evaluation_loader(load_split(root, test_csv, preprocess))
    num_batches = len(loader)

    model = BiModalModel(device=device, freeze_blocks=freeze_blocks)
    logit_scale = new_logit_scale(device)
    optimizer = build_optimizer(model, logit_scale, clip_lr=clip_lr, logit_scale_lr=logit_scale_lr,
                                device=device, clip_optimizer_mode=clip_optimizer_mode)
    total_steps = optimizer_updates_per_epoch(num_batches, accum_steps) * epochs
    scheduler = warmup_cosine_schedule(optimizer, total_steps, int(total_steps * warmup_ratio))
    output_path = os.path.join(root, output)
    best_score, best_epoch = float("-inf"), None
    print(f"[TI] {epochs} epochs, batch {batch_size} x {accum_steps}, CLIP lr {clip_lr:g}, "
          f"CLIP optimizer {clip_optimizer_mode}, seed {seed}")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, finite_steps = 0.0, 0
        progress = tqdm(enumerate(loader), total=num_batches, desc=f"Epoch {epoch}/{epochs}")
        for step, batch in progress:
            batch = batch.to(device)
            with torch.amp.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                v_i, v_t = model(batch)
                group_size = accumulation_group_size(step, num_batches, accum_steps)
                loss = contrastive_loss(v_i, v_t, logit_scale, label_smoothing) / group_size
            loss.backward()
            if (step + 1) % accum_steps == 0 or step + 1 == num_batches:
                optimizer_step(model, optimizer, scheduler, logit_scale)
            step_loss = loss.item() * group_size
            if np.isfinite(step_loss):
                total_loss += step_loss
                finite_steps += 1
            progress.set_postfix(loss=f"{step_loss:.4f}")

        recall = evaluate(model, eval_loader, device)
        score = recall["T2I"]["R@1"]
        print(f"Epoch {epoch}: loss {total_loss / max(1, finite_steps):.4f}, "
              f"1/tau {logit_scale.exp().item():.2f}, selection R@1 {score:.2f}\n{format_results(recall)}")
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(model, logit_scale, output_path, model_type="TI", training_config=config,
                            selection={"epoch": epoch, "directions": ["T2I"], "mean_r1": score,
                                       "recall": recall})
            print(f"Saved best checkpoint (epoch {epoch}) to {output_path}")

    print(f"Best Text->Image R@1 {best_score:.2f} at epoch {best_epoch}: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip_lr", type=float, default=5e-5)
    parser.add_argument("--logit_scale_lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--freeze_blocks", type=int, default=16,
                        help="Number of bottom ViT blocks kept frozen.")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--accum_steps", type=int, default=1)
    parser.add_argument("--no_augmentation", action="store_true")
    parser.add_argument("--clip_optimizer_mode", default="legacy_cuda_duplicate", choices=CLIP_OPTIMIZER_MODES)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="checkpoints/ti.pth")
    args = vars(parser.parse_args())
    args["use_augmentation"] = not args.pop("no_augmentation")
    train(**args)


if __name__ == "__main__":
    main()
