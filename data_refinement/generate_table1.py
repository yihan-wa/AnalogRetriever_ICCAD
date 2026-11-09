"""Assemble paper Table 1 (dataset quality before and after refinement) from the
checker outputs written by run_refinement.sh.

Usage:
    python data_refinement/generate_table1.py --original original_check.json \
        --stage1 stage1_check.json --stage2 stage2_check.json --output table1.json
"""
import argparse
import json
from pathlib import Path

ROWS = ("Original Masala-CHAI", "+ Stage 1 (Initial Repair)", "+ Stage 2 (Feedback Refinement)")


def rates(check_file):
    """(# triplets, compile %, DC pass %, # passing both) from a spice_checker.py output file."""
    details = json.loads(Path(check_file).read_text())["details"]
    total = len(details)
    compiled = sum(bool(item.get("compile_success")) for item in details)
    dc = sum(bool(item.get("dc_pass")) for item in details)
    both = sum(bool(item.get("compile_success") and item.get("dc_pass")) for item in details)
    return total, round(100 * compiled / total, 2), round(100 * dc / total, 2), both


def build_table(original, stage1, stage2):
    rows = []
    for stage, check_file in zip(ROWS, (original, stage1, stage2)):
        total, compile_rate, dc_rate, _ = rates(check_file)
        rows.append({"stage": stage, "triplets": total, "compile": compile_rate, "dc_pass": dc_rate})
    both = rates(stage2)[3]
    rows.append({"stage": "Passing both checks", "triplets": both, "compile": 100.0, "dc_pass": 100.0})
    return rows


def format_table(rows):
    lines = ["Table 1. Dataset quality before and after refinement.",
             f"{'Stage':<34} {'# Triplets':>10} {'Compile (%)':>12} {'DC Pass (%)':>12}"]
    lines += [f"{row['stage']:<34} {row['triplets']:>10,} {row['compile']:>12} {row['dc_pass']:>12}"
              for row in rows]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--original", required=True, help="spice_checker.py output for the raw dataset")
    parser.add_argument("--stage1", required=True, help="spice_checker.py output after Stage 1")
    parser.add_argument("--stage2", required=True, help="spice_checker.py output after Stage 2")
    parser.add_argument("--output", required=True, help="JSON output; a .txt table is written alongside")
    args = parser.parse_args()
    rows = build_table(args.original, args.stage1, args.stage2)
    table = format_table(rows)
    print(table)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"table": rows}, indent=2) + "\n")
    output.with_suffix(".txt").write_text(table + "\n")
    print(f"Saved {output} and {output.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
