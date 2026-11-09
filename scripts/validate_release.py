"""Check the committed split and, optionally, a downloaded dataset or checkpoint
against ``release_manifest.json``.

Usage:
    python scripts/validate_release.py [--data-dir Masala-Chai] [--checkpoint checkpoints/analogretriever.pth]
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_ids(train, test, total=6371, test_size=1000, seed=42):
    """The paper split: sorted IDs shuffled with Python seed 42, first 1,000 held out."""
    if len(train) != total - test_size or len(test) != test_size:
        raise ValueError("Unexpected train/test size")
    if len(set(train + test)) != total:
        raise ValueError("Duplicate or overlapping circuit IDs")
    expected = sorted(train + test)
    random.Random(seed).shuffle(expected)
    if test != expected[:test_size] or train != expected[test_size:]:
        raise ValueError(f"Split order differs from the seed-{seed} paper split")


def validate_data(data_dir, expected_ids):
    """Every circuit has exactly one readable image and nonempty caption and netlist files."""
    from PIL import Image

    data_dir = Path(data_dir)
    expected_ids = set(expected_ids)
    for folder, prefix in (("captions", "cap"), ("spice", "spice")):
        files = {int(path.stem[len(prefix):]): path for path in (data_dir / folder).glob(f"{prefix}*.txt")
                 if path.stem[len(prefix):].isdigit()}
        if set(files) != expected_ids:
            raise ValueError(f"{folder}: circuit IDs differ from the published split")
        for path in files.values():
            if not path.read_text(encoding="utf-8").strip():
                raise ValueError(f"Empty file: {path}")

    images = {}
    for path in (data_dir / "images").iterdir():
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"} and path.stem[3:].isdigit() \
                and path.stem.startswith("img"):
            images.setdefault(int(path.stem[3:]), []).append(path)
    if set(images) != expected_ids:
        raise ValueError("images: circuit IDs differ from the published split")
    for circuit, paths in images.items():
        if len(paths) != 1:
            raise ValueError(f"Circuit {circuit}: expected exactly one image, found {len(paths)}")
        try:
            with Image.open(paths[0]) as image:
                image.verify()
            with Image.open(paths[0]) as image:
                image.load()
        except (OSError, ValueError) as error:
            raise ValueError(f"Invalid image artifact: {paths[0]}") from error


def validate_checkpoint(path, manifest):
    path = Path(path)
    entry = next((item for item in manifest.get("checkpoints", {}).values()
                  if item["filename"].split("/")[-1] == path.name), None)
    if entry is None:
        raise ValueError(f"{path.name} is not a released checkpoint in release_manifest.json")
    if path.stat().st_size != entry["size_bytes"] or file_sha256(path) != entry["sha256"]:
        raise ValueError(f"{path} differs from the released checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "release_manifest.json").read_text())
    for name, expected in manifest["files_sha256"].items():
        if file_sha256(ROOT / name) != expected:
            raise ValueError(f"Checksum mismatch: {name}")
    train = [int(x) for x in (ROOT / "splits/train_ids.txt").read_text().split()]
    test = [int(x) for x in (ROOT / "splits/test_ids.txt").read_text().split()]
    validate_ids(train, test, manifest["dataset"]["triplets"], manifest["dataset"]["test"],
                 manifest["dataset"]["split_seed"])
    checked = ["split"]
    if args.data_dir:
        validate_data(args.data_dir, train + test)
        checked.append(f"dataset {args.data_dir}")
    if args.checkpoint:
        validate_checkpoint(args.checkpoint, manifest)
        checked.append(f"checkpoint {args.checkpoint}")
    print("Validated " + ", ".join(checked) + ".")


if __name__ == "__main__":
    main()
