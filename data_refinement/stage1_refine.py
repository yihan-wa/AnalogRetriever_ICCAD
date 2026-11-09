"""
Stage 1: repair mapped original netlists; generate only genuinely missing pairs.
Original captions are preserved for the subsequent description-refinement stage.
"""

import os
import json
import re
import time
import argparse
import shutil
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
try:
    from . import llm_client
    from .llm_client import call_llm
    from .pipeline_io import (load_prompts, atomic_json, copy_image,
                              quality_summary, restore_image, content_fingerprint,
                              load_progress, output_fingerprint, is_current)
except ImportError:  # Direct script execution.
    import llm_client
    from llm_client import call_llm
    from pipeline_io import (load_prompts, atomic_json, copy_image,
                             quality_summary, restore_image, content_fingerprint,
                             load_progress, output_fingerprint, is_current)
try:
    from .spice_checker import check_netlist
except ImportError:
    from spice_checker import check_netlist

# ── Prompts ──────────────────────────────────────────────────────────────────

SYSTEM_PROMPT, USER_PROMPT = load_prompts("stage1_prompt.md")
STAGE1_PROTOCOL_VERSION = "stage1-original-repair-v1"


def _check_record(checked):
    result = {key: checked.get(key, False if key.endswith(('success', 'pass')) else '')
              for key in ('compile_success', 'compile_errors', 'dc_pass', 'dc_errors')}
    result.update({key: checked[key] for key in ('raw_output', 'dc_output') if key in checked})
    return result


def parse_response(response: str) -> dict:
    """Parse LLM response to extract netlist and description."""
    netlist = ""
    description = ""

    # Extract netlist
    m = re.search(r'<NETLIST>\s*(.*?)\s*</NETLIST>', response, re.DOTALL)
    if m:
        netlist = m.group(1).strip()
        # Remove markdown code fences if present
        netlist = re.sub(r'^```\w*\n?', '', netlist)
        netlist = re.sub(r'\n?```$', '', netlist)
        netlist = netlist.strip()

    # Extract description
    m = re.search(r'<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>', response, re.DOTALL)
    if m:
        description = m.group(1).strip()

    return {"netlist": netlist, "description": description}


def process_one(image_path: str, idx: int, total: int, prompts=None) -> dict:
    """Generate an initial pair only for an image absent from the original mapping."""
    print(f"  [{idx+1}/{total}] Processing {os.path.basename(image_path)}...")

    system_prompt, user_prompt = prompts or load_prompts("stage1_prompt.md")
    response = call_llm(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_path=image_path,
        temperature=llm_client.TEMPERATURE,
        max_tokens=llm_client.MAX_TOKENS,
    )

    if not response:
        return {
            "success": False,
            "error": "LLM returned no response",
            "netlist": "",
            "description": "",
            "check": {"compile_success": False, "dc_pass": False},
        }

    parsed = parse_response(response)

    if not parsed["netlist"] or not parsed["description"]:
        return {
            "success": False,
            "error": "Failed to parse netlist or description from response",
            "netlist": "",
            "description": parsed.get("description", ""),
            "raw_response": response[:500],
            "check": {"compile_success": False, "dc_pass": False},
        }

    # Check compilation and DC simulation
    check = check_netlist(parsed["netlist"])

    return {
        "success": True,
        "netlist": parsed["netlist"],
        "description": parsed["description"],
        "check": _check_record(check),
    }


def process_existing(data_dir, entry, prompts):
    """Make one initial repair request with original text and real checker errors."""
    original_netlist = (Path(data_dir) / entry['original_spice']).read_text(encoding='utf-8')
    original_caption = (Path(data_dir) / entry['original_caption']).read_text(encoding='utf-8')
    checked = check_netlist(original_netlist)
    initial_check = _check_record(checked)
    system_prompt, template = prompts
    response = call_llm(
        system_prompt=system_prompt,
        user_prompt=template.format(netlist=original_netlist,
                                    check_report=json.dumps(initial_check, indent=2)),
        image_path=str(Path(data_dir) / entry['image']),
        temperature=llm_client.TEMPERATURE,
        max_tokens=llm_client.MAX_TOKENS,
    )
    repaired = parse_response(response or '')['netlist']
    result = {'success': bool(repaired), 'netlist': repaired,
              'description': original_caption, 'input_netlist': original_netlist,
              'input_caption': original_caption, 'initial_check': initial_check}
    if not repaired:
        result['error'] = 'Original-netlist repair API/format failure; rerun to retry'
        result['check'] = {'compile_success': False, 'dc_pass': False}
    else:
        final_check = check_netlist(repaired)
        result['check'] = _check_record(final_check)
    return result


def _source_file(data_dir, value, label):
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f'{label}: expected a nonempty relative path')
    path = (Path(data_dir) / value).resolve()
    if not path.is_relative_to(Path(data_dir).resolve()):
        raise ValueError(f'{label}: path leaves the source dataset: {value}')
    if not path.is_file() or not path.stat().st_size:
        raise ValueError(f'{label}: missing or empty file: {value}')
    return path


def build_image_list(data_dir: str, mapping_path: str) -> list:
    """Validate all original triplets before routing any image or making API calls."""
    data_dir = Path(data_dir).resolve()
    mapping_path = Path(mapping_path)
    if not mapping_path.is_file() or not mapping_path.stat().st_size:
        raise ValueError(f"Missing or empty original mapping: {mapping_path}")
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    if not isinstance(mapping, list):
        raise ValueError("Original mapping must be a JSON list")
    images_dir = data_dir / "images"
    if not images_dir.is_dir():
        raise ValueError(f"Missing images directory: {images_dir}")

    issues, images, image_ids = [], {}, {}
    for path in sorted(images_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        key = path.relative_to(data_dir).as_posix()
        match = re.fullmatch(r"img(\d+)\.(?:jpg|jpeg|png)", path.name, re.IGNORECASE)
        if not match:
            issues.append(f"Invalid image filename (expected img<ID>.jpg/jpeg/png): {key}")
            continue
        img_id = str(int(match.group(1)))
        if img_id in image_ids:
            issues.append(f"Duplicate image ID {img_id}: {image_ids[img_id]}, {key}")
        image_ids[img_id] = key
        try:
            _source_file(data_dir, key, "image")
        except ValueError as error:
            issues.append(str(error))
        images[key] = img_id
    if not images:
        issues.append("No supported images found")

    mapping_lookup, used_paths = {}, {"spice": set(), "caption": set()}
    for index, entry in enumerate(mapping):
        label = f"mapping[{index}]"
        if not isinstance(entry, dict):
            issues.append(f"{label}: expected an object with image/spice/caption")
            continue
        valid, paths = True, {}
        for field in ("image", "spice", "caption"):
            try:
                source = _source_file(data_dir, entry.get(field), f"{label}.{field}")
                paths[field] = source.relative_to(data_dir).as_posix()
                if field != "image" and not source.read_text(encoding="utf-8").strip():
                    raise ValueError(f"{label}.{field}: blank text: {entry[field]}")
            except (ValueError, OSError) as error:
                issues.append(str(error))
                valid = False
        if not valid:
            continue
        key = paths["image"]
        if key not in images:
            issues.append(f"{label}.image is not a supported images/ input: {key}")
        if key in mapping_lookup:
            issues.append(f"Duplicate image mapping: {key}")
        for field in ("spice", "caption"):
            if paths[field] in used_paths[field]:
                issues.append(f"Duplicate {field} mapping: {paths[field]}")
            used_paths[field].add(paths[field])
        mapping_lookup[key] = paths

    if issues:
        shown = "\n".join(f"  - {issue}" for issue in issues[:20])
        if len(issues) > 20:
            shown += f"\n  ... and {len(issues) - 20} more"
        raise ValueError(f"Stage 1 input preflight failed ({len(issues)} issue(s)); "
                         f"repair the original mapping/files before any API calls:\n{shown}")
    return [dict(image=key, img_id=images[key], has_original=key in mapping_lookup,
                 source_mode="repair_existing" if key in mapping_lookup else "generate_missing",
                 original_spice=mapping_lookup.get(key, {}).get("spice", ""),
                 original_caption=mapping_lookup.get(key, {}).get("caption", ""))
            for key in sorted(images, key=lambda key: int(images[key]))]


def route_counts(image_list):
    original = sum(entry["has_original"] for entry in image_list)
    return {"total_images": len(image_list), "repair_existing": original,
            "generate_missing": len(image_list) - original}


def _run_inputs(data_dir, image_list):
    """Freeze executable prompts/config and fingerprint each sample's original inputs."""
    prompt_pairs = {"repair_existing": load_prompts("stage1_repair_prompt.md"),
                    "generate_missing": load_prompts("stage1_prompt.md")}
    implementation = Path(__file__).parent
    shared_files = {name: implementation / name for name in
                    ("stage1_refine.py", "pipeline_io.py", "spice_checker.py", "llm_client.py")}
    shared_files.update({name: implementation / "prompts" / name for name in
                         ("stage1_repair_prompt.md", "stage1_prompt.md")})
    config = {"protocol_version": STAGE1_PROTOCOL_VERSION, "prompts": prompt_pairs,
              "model": llm_client.MODEL_NAME, "temperature": llm_client.TEMPERATURE,
              "max_tokens": llm_client.MAX_TOKENS}
    shared_fingerprint = content_fingerprint(shared_files, config)
    fingerprints = {}
    for entry in image_list:
        files = {"image": Path(data_dir) / entry["image"]}
        if entry["has_original"]:
            files.update({key: Path(data_dir) / entry[key]
                          for key in ("original_spice", "original_caption")})
        fingerprints[entry["image"]] = content_fingerprint(
            files, {"shared_fingerprint": shared_fingerprint, "source": entry})
    return prompt_pairs, config, fingerprints


def run_stage1(data_dir, mapping_path, output_dir, start=0, end=None, workers=8, delay=0.2):
    if start < 0 or (end is not None and end < start) or workers < 1 or delay < 0:
        raise ValueError("Invalid Stage 1 range, workers or delay")
    data_dir = Path(data_dir).resolve()
    image_list = build_image_list(data_dir, mapping_path)
    subset = image_list[start:end]
    output = Path(output_dir).resolve()
    if output == data_dir:
        raise ValueError("Stage 1 output directory must differ from the original dataset")
    progress_file = output / "stage1_progress.json"
    progress = load_progress(progress_file, STAGE1_PROTOCOL_VERSION)
    prompts, config, fingerprints = _run_inputs(data_dir, image_list)
    progress["configuration"] = config
    for folder in ("spice", "captions", "images"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    # Reuse only this protocol's unchanged inputs and unchanged complete outputs.
    for key, result in list(progress["completed"].items()):
        if key not in fingerprints:
            progress["completed"].pop(key)
            continue
        if result.get("input_fingerprint") == fingerprints[key]:
            restore_image(data_dir, output, result)
        if not is_current(output, result, fingerprints[key]):
            progress["failed"][key] = progress["completed"].pop(key)
            progress["failed"][key]["error"] = "Input/configuration or output content changed; rerun required"
    progress["failed"] = {key: value for key, value in progress["failed"].items() if key in fingerprints}
    atomic_json(progress_file, progress)
    lock = threading.Lock()

    def process(item):
        index, entry = item
        key, img_id = entry["image"], entry["img_id"]
        if key in progress["completed"]:
            return
        try:
            if entry["has_original"]:
                result = process_existing(data_dir, entry, prompts["repair_existing"])
            else:
                result = process_one(str(data_dir / key), index, len(subset),
                                     prompts["generate_missing"])
            result.update(image=key, img_id=img_id, has_original=entry["has_original"],
                          source_mode=entry["source_mode"], input_fingerprint=fingerprints[key],
                          original_spice=entry.get("original_spice", ""),
                          original_caption=entry.get("original_caption", ""),
                          source_inputs={field: str(data_dir / entry[field]) for field in
                                         ("image", "original_spice", "original_caption") if entry[field]},
                          caption_source="original" if entry["has_original"] else "generated")
            if not result.get("success") or not result.get("netlist", "").strip() or not result.get("description", "").strip():
                result["success"] = False
                result.setdefault("error", "Missing netlist or description")
            else:
                result["output_spice"] = f"spice/spice{img_id}.txt"
                result["output_caption"] = f"captions/cap{img_id}.txt"
                (output / result["output_spice"]).write_text(result["netlist"], encoding="utf-8")
                (output / result["output_caption"]).write_text(result["description"], encoding="utf-8")
                copy_image(data_dir, output, key)
                result["output_fingerprint"] = output_fingerprint(output, result)
        except Exception as error:
            result = {"success": False, "image": key, "error": str(error),
                      "source_mode": entry["source_mode"], "input_fingerprint": fingerprints[key]}
        with lock:
            if is_current(output, result, fingerprints[key]):
                progress["completed"][key] = result
                progress["failed"].pop(key, None)
            else:
                progress["failed"][key] = result
            progress["summary"] = quality_summary(progress["completed"].values())
            atomic_json(progress_file, progress)
        if delay:
            time.sleep(delay)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process, enumerate(subset)))
    successful = [progress["completed"][entry["image"]] for entry in image_list
                  if entry["image"] in progress["completed"]
                  and is_current(output, progress["completed"][entry["image"]], fingerprints[entry["image"]])]
    mapping = [{"image": result["image"], "spice": result["output_spice"],
                "caption": result["output_caption"]} for result in successful]
    summary = quality_summary(successful)
    summary.update(**route_counts(image_list), failed=len(progress["failed"]),
                   protocol_version=STAGE1_PROTOCOL_VERSION)
    progress["summary"] = summary
    atomic_json(progress_file, progress)
    atomic_json(output / "data_mapping.json", mapping)
    atomic_json(output / "stage1_summary.json", summary)
    print(f"Stage 1: {len(mapping)} complete triplets; {summary['failed']} retryable failures")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 1: repair originals and generate missing pairs")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--validate-only", action="store_true",
                        help="Validate original triplets and report route counts without API calls")
    parser.add_argument("--expected-original", type=int)
    parser.add_argument("--expected-generated", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    if not args.validate_only and not args.output_dir:
        parser.error("--output-dir is required unless --validate-only is used")
    try:
        counts = route_counts(build_image_list(args.data_dir, args.mapping))
        for expected, key in ((args.expected_original, "repair_existing"),
                              (args.expected_generated, "generate_missing")):
            if expected is not None and counts[key] != expected:
                raise ValueError(f"Expected {expected} {key} samples, found {counts[key]}; "
                                 "verify the original mapping and image inventory")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if args.validate_only:
        print(json.dumps(counts, indent=2))
        return
    if not shutil.which("ngspice"):
        parser.error("ngspice is required for Stage 1 checks; install it or add it to PATH")
    summary = run_stage1(args.data_dir, args.mapping, args.output_dir, args.start,
                         args.end, args.workers, args.delay)
    if summary["failed"]:
        raise SystemExit("Stage 1 has retryable failures; rerun to complete them before Stage 2")


if __name__ == "__main__":
    main()
