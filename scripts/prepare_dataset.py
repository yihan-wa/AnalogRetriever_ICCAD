"""Write train_data.csv and test_data.csv for the paper split (5,371 / 1,000 triplets).

Rows follow the ordered circuit IDs in splits/; every circuit must have an
image, a caption and a netlist under the data directory.

Usage:
    python scripts/prepare_dataset.py [--data-dir Masala-Chai]
"""
import argparse
import csv
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analogretriever.paths import repository_root


def prepare_dataset(data_dir, output_dir, split_dir):
    data_dir, output_dir, split_dir = (Path(path).resolve() for path in (data_dir, output_dir, split_dir))
    splits = {name: [int(value) for value in (split_dir / f"{name}_ids.txt").read_text().split()]
              for name in ("train", "test")}
    all_ids = splits["train"] + splits["test"]
    if not all(splits.values()) or len(set(all_ids)) != len(all_ids):
        raise ValueError("Split IDs must be nonempty, unique and disjoint")
    spice_dir = data_dir / "spice"
    image_files = {}
    for path in (data_dir / "images").iterdir():
        if path.is_file() and path.suffix.lower() in {".jpg", ".png", ".jpeg"}:
            image_files.setdefault(path.stem, []).append(path)
    records = {}
    # Validate every row before writing either CSV.
    for cid in all_ids:
        caption_path = data_dir / "captions" / f"cap{cid}.txt"
        spice_path = spice_dir / f"spice{cid}.txt"
        existing_images = image_files.get(f"img{cid}", [])
        if len(existing_images) != 1:
            raise ValueError(f"Circuit {cid}: expected exactly one image, found {len(existing_images)}")
        image_path = existing_images[0]
        for path in (image_path, caption_path, spice_path):
            if not path.is_file() or not path.stat().st_size:
                raise FileNotFoundError(f"Missing/empty modality for committed circuit {cid}: {path}")
        caption = caption_path.read_text(encoding="utf-8").strip()
        if not caption or not spice_path.read_text(encoding="utf-8").strip():
            raise ValueError(f"Circuit {cid}: caption and netlist must be nonblank")
        records[cid] = {"id": cid, "image": os.path.relpath(image_path, output_dir),
                        "spice": os.path.relpath(spice_path, output_dir), "caption": caption}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, ids in splits.items():
        destination = output_dir / f"{name}_data.csv"
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("id", "image", "spice", "caption"))
            writer.writeheader()
            writer.writerows(records[cid] for cid in ids)
        temporary.replace(destination)
        print(f"{name}: {len(ids)} committed IDs -> {destination}")
    return splits


def main():
    root = Path(repository_root())
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=root / "Masala-Chai")
    args = parser.parse_args()
    prepare_dataset(args.data_dir, root, root / "splits")


if __name__ == "__main__":
    main()
