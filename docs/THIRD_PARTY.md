# Third-party components

The code in this repository is released under the MIT License. The models,
tools and data below keep their own licenses and terms of use.

| Component | Use | Source |
|---|---|---|
| OpenAI CLIP ViT-L/14 | Schematic and description encoders | [openai/CLIP](https://github.com/openai/CLIP) |
| PyTorch, PyTorch Geometric | Training and graph layers | [pytorch](https://github.com/pytorch/pytorch), [pyg-team/pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) |
| Qwen3-Embedding-0.6B | Functional clusters; NetTAG baseline | [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) |
| Qwen2.5-7B-Instruct | CROP and ChatLS baseline summaries | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| GPT-5.4 via OpenRouter | Dataset refinement | [OpenRouter](https://openrouter.ai/) |
| Ngspice | Netlist compilation and DC checks | [ngspice.sourceforge.io](https://ngspice.sourceforge.io/) |
| Masala-CHAI | Source schematics, netlists and captions | [Bhandari et al., arXiv:2411.14299](https://arxiv.org/abs/2411.14299), [repository](https://github.com/jitendra-bhandari/Masala-CHAI) |

The CROP ([Pan et al., ICCAD 2025](https://doi.org/10.1109/ICCAD66269.2025.11240643)),
ChatLS ([Zheng et al., DAC 2025](https://doi.org/10.1109/DAC63849.2025.11132969)) and
NetTAG ([Fang et al., DAC 2025](https://doi.org/10.1109/DAC63849.2025.11133349))
baselines are our re-implementations of their netlist representations for
analog circuits (see the README); they do not include code from the original
projects.
