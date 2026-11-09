# AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval

[Paper (ICCAD 2026)](https://doi.org/10.1145/3831252.3834253) ·
[arXiv](https://arxiv.org/abs/2604.23195) ·
[Dataset](https://huggingface.co/datasets/eulermaxwell/AnalogRetriever-data) ·
[Checkpoints](https://huggingface.co/eulermaxwell/AnalogRetriever) ·
[Demo](https://huggingface.co/spaces/eulermaxwell/AnalogRetriever) ·
[中文说明](README_zh-CN.md)

AnalogRetriever retrieves analog circuits across three modalities: schematic
images, functional descriptions and SPICE netlists. Schematics and
descriptions are encoded with CLIP ViT-L/14, netlists with a port-aware
relational graph convolutional network (RGCN), and the three modalities are
aligned in one 768-dimensional embedding space by curriculum contrastive
learning. This repository contains the code for the paper's dataset
refinement (Table 1), training, and retrieval evaluation (Table 2).

## Installation

Python 3.10+ and a CUDA GPU are required for training; evaluation also runs on CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
# Install PyTorch for your CUDA version first: https://pytorch.org/get-started/locally/
bash scripts/install_env.sh        # requirements.txt + editable install of this package
```

[Ngspice](https://ngspice.sourceforge.io/) is needed only for data refinement.

## Data

The refined dataset has 6,371 (schematic, description, netlist) triplets. The
paper split shuffles the sorted circuit IDs with seed 42 and holds out the first
1,000; the ordered IDs are in [`splits/`](splits).

```bash
hf download eulermaxwell/AnalogRetriever-data --repo-type dataset \
    --revision 0c053b6580315d8a89c629d67994a6da4575a17f \
    --include "iccad2026/*" --local-dir hf_data
mv hf_data/iccad2026 Masala-Chai
python3 scripts/validate_release.py --data-dir Masala-Chai
python3 scripts/prepare_dataset.py          # writes train_data.csv (5,371) and test_data.csv (1,000)
```

## Evaluate a released checkpoint

```bash
hf download eulermaxwell/AnalogRetriever analogretriever.pth analogretriever.pth.metadata.json --local-dir checkpoints
python3 scripts/validate_release.py --checkpoint checkpoints/analogretriever.pth   # optional integrity check
python3 -m analogretriever.evaluation.evaluate_retrieval --checkpoint checkpoints/analogretriever.pth
```

The CLIP weights are part of the checkpoint, so no other download is needed.
Each evaluation draws ten 200-item galleries from the 1,000 held-out triplets
and reports Recall@1/5/10 for all six directions. Add `--device cpu` to run
without a GPU.

To retrieve circuits with your own query, encode it with the model and rank
the encoded library by cosine similarity:

```python
import clip, torch
from analogretriever.checkpoints import load_checkpoint
from analogretriever.dataset import parse_netlist

model, _, _ = load_checkpoint("checkpoints/analogretriever.pth", device="cpu")
with torch.no_grad():
    query = model.text_encoder(clip.tokenize(["a two-stage op-amp with Miller compensation"]))
    x_type, x_continuous, edge_index, edge_type = parse_netlist(open("Masala-Chai/spice/spice1.txt").read())
    netlist = model.netlist_encoder(x_type, x_continuous, edge_index,
                                    torch.zeros(len(x_type), dtype=torch.long), edge_type)
print((query @ netlist.t()).item())   # cosine similarity
```

## Reproduce Table 2

```bash
bash scripts/run_experiments.sh
```

The script prepares the split, builds the functional clusters, evaluates the
four zero-shot baselines, trains TI, TIC (GCN), TIC (RGCN) and
AnalogRetriever, evaluates every checkpoint and writes `results/table2.md`.
The released checkpoints were trained on one NVIDIA H200 (PyTorch 2.15, CUDA 13.2) with bf16
autocast. The individual steps are:

| Step | Command |
|---|---|
| Functional clusters (K-means, K=30, Qwen3-Embedding-0.6B) | `python3 scripts/build_clusters.py` |
| AnalogRetriever (three-phase curriculum) | `python3 -m analogretriever.training.train_curriculum` |
| TIC (RGCN) / TIC (GCN) | `python3 -m analogretriever.training.train_trimodal --conv rgcn` / `--conv gcn` |
| TI (bi-modal) | `python3 -m analogretriever.training.train_bimodal` |
| Zero-shot baselines | `python3 -m analogretriever.evaluation.evaluate_{clip,crop,chatls,nettag}` |
| Evaluate a checkpoint | `python3 -m analogretriever.evaluation.evaluate_retrieval --checkpoint <path>` |
| Assemble Table 2 | `python3 scripts/generate_table2.py` |

The clusters used for the released checkpoint are in
[`artifacts/train_clusters.json`](artifacts/train_clusters.json); pass
`--clusters artifacts/train_clusters.json` to `train_curriculum` to reuse them.

The defaults of every training script are the paper settings (Sec. 4.2):
20 epochs (6 / 2 / 12 for the three curriculum phases), batch size 256,
AdamW with learning rate 5e-5 for CLIP and 5e-4 for the graph encoder,
bottom 16 of 24 ViT blocks frozen, label smoothing 0.1, learnable temperature
initialised to 0.07, hard-negative ratio 0.05 → 0.3 and seed 42. Checkpoints
are selected by held-out R@1 averaged over Image→Code, Text→Image and
Text→Code (Text→Image for TI).

CLIP's image and text encoders share one CLIP model, and by default
(`--clip_optimizer_mode legacy_cuda_duplicate`) its trainable parameters are
registered with AdamW once per encoder, as in the training of the released
checkpoints; on CUDA each shared parameter then receives two AdamW updates
per step. `--clip_optimizer_mode standard` registers every parameter once.

### Results

This release corrects two issues of the experiment code: the per-epoch
held-out evaluation no longer re-seeds PyTorch's global random number
generator, which had fixed the batch order and augmentation of the TI and TIC
training runs after the first epoch; and SPICE values with unit suffixes
(e.g. `1uF`) and the capacitance feature are now converted correctly. The
released checkpoints and the table below were obtained with this code.

Recall@1 (%) on the held-out set (mean of ten 200-item galleries). I:
schematic image, T: description, C: netlist. Recall@5/10 are in
[`results/table2.md`](results/table2.md).

| Model | I→C | T→I | T→C | C→I | I→T | C→T | Avg |
|---|---|---|---|---|---|---|---|
| CLIP | 1.2 | 2.2 | 2.9 | 2.9 | 2.7 | 3.2 | 2.5 |
| CROP | 2.1 | 2.2 | 9.1 | 2.3 | 2.7 | 8.5 | 4.5 |
| ChatLS | 1.5 | 2.2 | 11.8 | 1.8 | 2.7 | 8.3 | 4.7 |
| NetTAG | 0.4 | 2.2 | 0.3 | 0.4 | 2.7 | 0.1 | 1.1 |
| TI (Bi-Modal) | – | 76.8 | – | – | 76.1 | – | – |
| TIC (GCN) | 69.7 | 75.6 | 74.0 | 67.5 | 76.4 | 70.2 | 72.3 |
| TIC (RGCN) | 71.9 | 76.0 | **75.7** | 69.2 | 76.5 | **72.4** | 73.6 |
| **AnalogRetriever** | **72.9** | **78.5** | 74.6 | **70.7** | **77.3** | 71.7 | **74.3** |

### Baselines

All baselines are applied zero-shot and use the off-the-shelf CLIP for
schematics and descriptions, so they share the same T→I and I→T recall. They
differ in how a netlist is represented:

| Baseline | Netlist representation |
|---|---|
| CLIP | Raw SPICE text (first 300 characters) encoded by the CLIP text encoder. |
| CROP | Netlist summarized by Qwen2.5-7B-Instruct, summary encoded by CLIP. |
| ChatLS | Device/net statistics and the netlist summarized by Qwen2.5-7B-Instruct, summary encoded by CLIP. |
| NetTAG | Per-device text attributes encoded with Qwen3-Embedding-0.6B and mean-pooled. |

CROP, ChatLS and NetTAG were proposed for other circuit domains; these are
our analog adaptations of their netlist representations, not the original
systems. Summaries use greedy decoding and are cached in
[`artifacts/`](artifacts). The NetTAG embedding (1024-d) is mapped to the
768-d CLIP space by a fixed random projection that only matches
dimensionality, so its code-involved directions are an unaligned control.

## Dataset refinement (Table 1)

[`data_refinement/`](data_refinement) contains the two-stage pipeline of
Sec. 4.1 that repairs Masala-CHAI with GPT-5.4 and Ngspice feedback. See
[`data_refinement/README.md`](data_refinement/README.md) for inputs, usage
and the provenance of the released dataset.

## Code and paper

| Paper | Code |
|---|---|
| Sec. 3.2 netlist graph (20 relation types, Eq. 1) | [`dataset.py`](src/analogretriever/dataset.py) |
| Sec. 3.2 encoders (Eq. 2–3) | [`encoders.py`](src/analogretriever/encoders.py) |
| Sec. 3.3 tri-modal contrastive loss (Eq. 4–5) | [`training/common.py`](src/analogretriever/training/common.py) |
| Sec. 3.4 three-phase curriculum (Eq. 6) | [`training/train_curriculum.py`](src/analogretriever/training/train_curriculum.py), [`scripts/build_clusters.py`](scripts/build_clusters.py) |
| Sec. 4.1 dataset refinement, Table 1 | [`data_refinement/`](data_refinement) |
| Sec. 4.2 evaluation protocol | [`evaluation/protocol.py`](src/analogretriever/evaluation/protocol.py) |
| Table 2 baselines | [`evaluation/evaluate_{clip,crop,chatls,nettag}.py`](src/analogretriever/evaluation) |
| Table 2 ablations | [`training/train_bimodal.py`](src/analogretriever/training/train_bimodal.py), [`training/train_trimodal.py`](src/analogretriever/training/train_trimodal.py) |

The retrieval-augmented generation experiments with AnalogCoder (Sec. 4.4,
Table 3) are not part of this repository.

Tests run with `python3 -m unittest discover -s tests`; tests that need
PyTorch, PyG or Ngspice are skipped when those are not installed.

## Citation

```bibtex
@inproceedings{wang2026analogretriever,
  title     = {AnalogRetriever: Learning Cross-Modal Representations for Analog Circuit Retrieval},
  author    = {Wang, Yihan and Li, Lei and Lai, Yao and Wang, Jing and Lu, Yan},
  booktitle = {IEEE/ACM International Conference on Computer-Aided Design (ICCAD '26)},
  year      = {2026},
  address   = {San Jose, CA, USA},
  publisher = {ACM},
  doi       = {10.1145/3831252.3834253}
}
```

## License

The code is released under the [MIT License](LICENSE). The dataset is derived
from [Masala-CHAI](https://github.com/jitendra-bhandari/Masala-CHAI) and keeps
its upstream terms; see [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md) for the
models and tools used.
