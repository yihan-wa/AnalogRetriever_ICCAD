"""Assemble paper Table 2 from the result files written by scripts/run_experiments.sh.

Usage:
    python scripts/generate_table2.py [--results-dir results]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Table 2 rows: (model, result file); zero-shot baselines first.
ROWS = (
    ("CLIP", "clip.json"), ("CROP", "crop.json"), ("ChatLS", "chatls.json"), ("NetTAG", "nettag.json"),
    ("TI (Bi-Modal)", "ti.json"), ("TIC (GCN)", "tic_gcn.json"), ("TIC (RGCN)", "tic_rgcn.json"),
    ("AnalogRetriever", "analogretriever.json"),
)
COLUMN_ORDER = ("I2C", "T2I", "T2C", "C2I", "I2T", "C2T")
DIRECTIONS = COLUMN_ORDER
KS = ("R@1", "R@5", "R@10")


def average_recall_at_1(results):
    """Mean R@1 over the six directions, or None if any direction is missing."""
    if results is None or any(name not in results for name in DIRECTIONS):
        return None
    return sum(results[name]["R@1"] for name in DIRECTIONS) / len(DIRECTIONS)


def fmt(value):
    return "–" if value is None else f"{value:.1f}"


def table_rows(results_dir):
    rows = []
    for model, filename in ROWS:
        path = Path(results_dir) / filename
        results = json.loads(path.read_text()) if path.is_file() else None
        cells = [fmt(results[d][k]) if results and d in results else "–" for d in COLUMN_ORDER for k in KS]
        rows.append((model, results, cells + [fmt(average_recall_at_1(results))]))
    return rows


def markdown(rows):
    header = "| Model | " + " | ".join(f"{d[0]}→{d[2]} {k}" for d in COLUMN_ORDER for k in KS) + " | Avg R@1 |"
    lines = [header, "|---" * (len(COLUMN_ORDER) * len(KS) + 2) + "|"]
    lines += [f"| {model} | " + " | ".join(cells) + " |" for model, _, cells in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", default=ROOT / "results", type=Path)
    args = parser.parse_args()
    rows = table_rows(args.results_dir)
    missing = [model for model, results, _ in rows if results is None]
    if missing:
        print("Missing results (shown as –): " + ", ".join(missing), file=sys.stderr)
    print("Table 2. Cross-modal retrieval on the held-out set (Recall@K, %).")
    print(markdown(rows))
    combined = {model: results for model, results, _ in rows}
    output = args.results_dir / "table2.json"
    output.write_text(json.dumps(combined, indent=2) + "\n")
    (args.results_dir / "table2.md").write_text(markdown(rows) + "\n")
    print(f"Saved {output} and {args.results_dir / 'table2.md'}")


if __name__ == "__main__":
    main()
