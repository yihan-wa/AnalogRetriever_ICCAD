# AnalogRetriever：面向模拟电路检索的跨模态表示学习

[论文（ICCAD 2026）](https://doi.org/10.1145/3831252.3834253) ·
[arXiv](https://arxiv.org/abs/2604.23195) ·
[数据集](https://huggingface.co/datasets/eulermaxwell/AnalogRetriever-data) ·
[模型权重](https://huggingface.co/eulermaxwell/AnalogRetriever) ·
[在线演示](https://huggingface.co/spaces/eulermaxwell/AnalogRetriever) ·
[English](README.md)

AnalogRetriever 在原理图、功能描述和 SPICE 网表三种模态之间检索模拟电路：原理图和描述由
CLIP ViT-L/14 编码，网表由端口感知的关系图卷积网络（RGCN）编码，三种模态通过课程式对比学习
对齐到同一个 768 维嵌入空间。本仓库包含论文中数据精炼（表 1）、训练与检索评估（表 2）的代码。

## 安装

训练需要 Python 3.10+ 和 CUDA GPU，评估也可以在 CPU 上运行。

```bash
python3 -m venv .venv && source .venv/bin/activate
# 先按 CUDA 版本安装 PyTorch：https://pytorch.org/get-started/locally/
bash scripts/install_env.sh        # 安装 requirements.txt，并以可编辑方式安装本项目
```

只有数据精炼需要 [Ngspice](https://ngspice.sourceforge.io/)。

## 数据

精炼后的数据集包含 6,371 组（原理图、描述、网表）三元组。论文划分将电路 ID 排序后用 seed 42
打乱，前 1,000 组作为留出集；有序 ID 见 [`splits/`](splits)。

```bash
hf download eulermaxwell/AnalogRetriever-data --repo-type dataset \
    --revision 0c053b6580315d8a89c629d67994a6da4575a17f \
    --include "iccad2026/*" --local-dir hf_data
mv hf_data/iccad2026 Masala-Chai
python3 scripts/validate_release.py --data-dir Masala-Chai
python3 scripts/prepare_dataset.py          # 生成 train_data.csv（5,371）和 test_data.csv（1,000）
```

## 评估发布的权重

```bash
hf download eulermaxwell/AnalogRetriever analogretriever.pth analogretriever.pth.metadata.json --local-dir checkpoints
python3 scripts/validate_release.py --checkpoint checkpoints/analogretriever.pth   # 可选：完整性校验
python3 -m analogretriever.evaluation.evaluate_retrieval --checkpoint checkpoints/analogretriever.pth
```

权重文件中已包含 CLIP 参数，无需另外下载。评估从 1,000 组留出样本中抽取 10 次 200 条候选库，
报告六个方向的 Recall@1/5/10。加 `--device cpu` 可在无 GPU 时运行。用自定义查询检索的示例见
[README.md](README.md#evaluate-a-released-checkpoint)。

## 复现表 2

```bash
bash scripts/run_experiments.sh
```

脚本依次完成数据划分、功能聚类、四个零样本基线评估，训练 TI、TIC（GCN）、TIC（RGCN）和
AnalogRetriever，评估所有权重并生成 `results/table2.md`。发布的权重在单张 NVIDIA H200（PyTorch 2.15、CUDA 13.2）
上以 bf16 autocast 训练。
各步骤命令：

| 步骤 | 命令 |
|---|---|
| 功能聚类（K-means，K=30，Qwen3-Embedding-0.6B） | `python3 scripts/build_clusters.py` |
| AnalogRetriever（三阶段课程训练） | `python3 -m analogretriever.training.train_curriculum` |
| TIC（RGCN）/ TIC（GCN） | `python3 -m analogretriever.training.train_trimodal --conv rgcn` / `--conv gcn` |
| TI（双模态） | `python3 -m analogretriever.training.train_bimodal` |
| 零样本基线 | `python3 -m analogretriever.evaluation.evaluate_{clip,crop,chatls,nettag}` |
| 评估权重 | `python3 -m analogretriever.evaluation.evaluate_retrieval --checkpoint <路径>` |
| 生成表 2 | `python3 scripts/generate_table2.py` |

发布权重所用的聚类结果见 [`artifacts/train_clusters.json`](artifacts/train_clusters.json)，训练时传入
`--clusters artifacts/train_clusters.json` 即可复用。

各训练脚本的默认参数即论文设置（第 4.2 节）：20 个 epoch（三个课程阶段分别 6 / 2 / 12），
batch size 256，AdamW 学习率 CLIP 5e-5、图编码器 5e-4，冻结 ViT 24 层中底部 16 层，标签平滑 0.1，
可学习温度初始为 0.07，难负样本比例 0.05 → 0.3，seed 42。权重按留出集上 Image→Code、Text→Image、
Text→Code 的平均 R@1 选择（TI 按 Text→Image）。

CLIP 的图像与文本编码器共享同一个 CLIP 模型。默认设置
（`--clip_optimizer_mode legacy_cuda_duplicate`）与发布权重的训练一致：可训练参数按两个编码器各登记
一次，在 CUDA 上每个共享参数每步接受两次 AdamW 更新。`--clip_optimizer_mode standard` 则只登记一次。

### 结果

本次发布修正了实验代码中的两个问题：训练过程中每个 epoch 的留出集评估不再重置 PyTorch 全局随机数
（此前会使 TI 和 TIC 训练从第二个 epoch 起使用相同的批次顺序与数据增强）；带单位后缀的 SPICE 数值
（如 `1uF`）及电容特征的单位换算已修正。发布的权重和下表均由本代码得到。

留出集上的 Recall@1（%，10 次 200 条候选库的平均）。I：原理图，T：描述，C：网表。Recall@5/10 见
[`results/table2.md`](results/table2.md)。

| 模型 | I→C | T→I | T→C | C→I | I→T | C→T | 平均 |
|---|---|---|---|---|---|---|---|
| CLIP | 1.2 | 2.2 | 2.9 | 2.9 | 2.7 | 3.2 | 2.5 |
| CROP | 2.1 | 2.2 | 9.1 | 2.3 | 2.7 | 8.5 | 4.5 |
| ChatLS | 1.5 | 2.2 | 11.8 | 1.8 | 2.7 | 8.3 | 4.7 |
| NetTAG | 0.4 | 2.2 | 0.3 | 0.4 | 2.7 | 0.1 | 1.1 |
| TI（双模态） | – | 76.8 | – | – | 76.1 | – | – |
| TIC（GCN） | 69.7 | 75.6 | 74.0 | 67.5 | 76.4 | 70.2 | 72.3 |
| TIC（RGCN） | 71.9 | 76.0 | **75.7** | 69.2 | 76.5 | **72.4** | 73.6 |
| **AnalogRetriever** | **72.9** | **78.5** | 74.6 | **70.7** | **77.3** | 71.7 | **74.3** |

### 基线

所有基线均为零样本，原理图和描述都用原始 CLIP 编码，因此 T→I、I→T 召回相同；区别在于网表的表示方式：

| 基线 | 网表表示 |
|---|---|
| CLIP | 原始 SPICE 文本（前 300 个字符）经 CLIP 文本编码器编码。 |
| CROP | Qwen2.5-7B-Instruct 生成网表摘要，再由 CLIP 编码。 |
| ChatLS | 器件/net 统计信息和网表一起由 Qwen2.5-7B-Instruct 生成摘要，再由 CLIP 编码。 |
| NetTAG | 逐器件文本属性经 Qwen3-Embedding-0.6B 编码后取均值。 |

CROP、ChatLS 和 NetTAG 原本面向其他电路领域，这里是我们针对模拟电路对其网表表示的适配，并非原系统。
摘要采用贪心解码并缓存在 [`artifacts/`](artifacts)。NetTAG 的 1024 维嵌入通过固定随机投影映射到 768 维
CLIP 空间，仅用于匹配维度，因此其涉及网表的方向是未对齐的对照。

## 数据精炼（表 1）

[`data_refinement/`](data_refinement) 实现第 4.1 节的两阶段流程，利用 GPT-5.4 和 Ngspice 反馈修复
Masala-CHAI。输入、用法以及已发布数据集的生成说明见
[`data_refinement/README.md`](data_refinement/README.md)。

## 代码与论文对应

| 论文 | 代码 |
|---|---|
| 3.2 节 网表图（20 种关系，式 1） | [`dataset.py`](src/analogretriever/dataset.py) |
| 3.2 节 编码器（式 2–3） | [`encoders.py`](src/analogretriever/encoders.py) |
| 3.3 节 三模态对比损失（式 4–5） | [`training/common.py`](src/analogretriever/training/common.py) |
| 3.4 节 三阶段课程训练（式 6） | [`training/train_curriculum.py`](src/analogretriever/training/train_curriculum.py)、[`scripts/build_clusters.py`](scripts/build_clusters.py) |
| 4.1 节 数据精炼、表 1 | [`data_refinement/`](data_refinement) |
| 4.2 节 评估协议 | [`evaluation/protocol.py`](src/analogretriever/evaluation/protocol.py) |
| 表 2 基线 | [`evaluation/evaluate_{clip,crop,chatls,nettag}.py`](src/analogretriever/evaluation) |
| 表 2 消融 | [`training/train_bimodal.py`](src/analogretriever/training/train_bimodal.py)、[`training/train_trimodal.py`](src/analogretriever/training/train_trimodal.py) |

与 AnalogCoder 结合的检索增强生成实验（4.4 节、表 3）不在本仓库中。

测试命令为 `python3 -m unittest discover -s tests`；未安装 PyTorch、PyG 或 Ngspice 时，相应测试会被跳过。

## 局限与后续工作

AnalogRetriever 是跨模态模拟电路检索的初步尝试，当前框架仍有诸多不足。我们正在持续优化和改进，包括但不限于：

- **数据**：在现有 6,371 组以教科书电路为主的数据基础上扩大规模、提升多样性，例如覆盖混合信号、射频和电源管理电路。
- **方法**：改进网表表示、编码器和训练策略。
- **评测**：采用更合理、更有说服力的评测体系，例如按电路类别划分数据、将测试集与模型选择分开、按电路功能而非严格配对判断检索相关性，以及报告多次训练的结果。

如有任何问题、错误报告或建议，欢迎提交 [issue](https://github.com/yihan-wa/AnalogRetriever_ICCAD/issues) 与我们交流，我们会积极改进代码和文档。

## 引用

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

## 许可

代码采用 [MIT 许可](LICENSE)。数据集源自 [Masala-CHAI](https://github.com/jitendra-bhandari/Masala-CHAI)，
沿用其上游条款；所用模型和工具见 [docs/THIRD_PARTY.md](docs/THIRD_PARTY.md)。
