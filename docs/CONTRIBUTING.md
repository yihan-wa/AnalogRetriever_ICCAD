# Contributing

Questions and bug reports are welcome as GitHub issues. Please include the
commit, Python/PyTorch/CUDA versions, the command you ran and the full error
message, and remove credentials from shared logs.

Before opening a pull request, run

```bash
python3 -m unittest discover -s tests
```

Changes to graph construction or the encoders change the meaning of trained
weights: bump `GRAPH_SCHEMA_VERSION` in `src/analogretriever/dataset.py` and
report retrieval results with the protocol in `evaluation/protocol.py`.
