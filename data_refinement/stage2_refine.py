"""
Stage 2: LLM-based Netlist Repair & Description Refinement.
  - Iteratively repair netlists that failed compilation/DC simulation (up to N retries).
  - Preserve Stage 1 successes; refine original captions only after failed netlists are repaired.
"""

import json
import re
import time
import argparse
import threading
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
try:
    from . import llm_client
    from .llm_client import call_llm
    from .pipeline_io import (load_prompts, atomic_json, output_ready, copy_image,
                              quality_summary, restore_image, content_fingerprint,
                              load_progress, output_fingerprint, is_current)
except ImportError:  # Direct script execution.
    import llm_client
    from llm_client import call_llm
    from pipeline_io import (load_prompts, atomic_json, output_ready, copy_image,
                             quality_summary, restore_image, content_fingerprint,
                             load_progress, output_fingerprint, is_current)
try:
    from .spice_checker import check_netlist
    from .stage1_refine import STAGE1_PROTOCOL_VERSION
except ImportError:
    from spice_checker import check_netlist
    from stage1_refine import STAGE1_PROTOCOL_VERSION

MAX_REPAIR_ITERATIONS = 5
PROTOCOL_VERSION = "stage2-feedback-repair-v1"

# ── Stage 2a: Netlist Repair Prompts ─────────────────────────────────────────

REPAIR_SYSTEM_PROMPT, REPAIR_USER_TEMPLATE = load_prompts("stage2_netlist_repair_prompt.md")


# ── Stage 2b: Description Refinement Prompts ─────────────────────────────────

DESC_SYSTEM_PROMPT, DESC_USER_TEMPLATE = load_prompts("stage2_description_refinement_prompt.md")



def parse_netlist_response(response: str) -> str:
    """Extract netlist from LLM response."""
    if not response:
        return ""
    m = re.search(r'<NETLIST>\s*(.*?)\s*</NETLIST>', response, re.DOTALL)
    if m:
        netlist = m.group(1).strip()
        netlist = re.sub(r'^```\w*\n?', '', netlist)
        netlist = re.sub(r'\n?```$', '', netlist)
        return netlist.strip()
    return ""


def parse_description_response(response: str) -> str:
    """Extract description from LLM response."""
    if not response:
        return ""
    m = re.search(r'<DESCRIPTION>\s*(.*?)\s*</DESCRIPTION>', response, re.DOTALL)
    if m:
        return m.group(1).strip()
    return ""


def repair_netlist(netlist: str, image_path: str, max_iters: int = MAX_REPAIR_ITERATIONS,
                   initial_check: dict = None) -> dict:
    """Check the input and make at most max_iters feedback-guided repairs."""
    if not 0 <= max_iters <= MAX_REPAIR_ITERATIONS:
        raise ValueError(f"max_iters must be between 0 and {MAX_REPAIR_ITERATIONS}")
    current_netlist = netlist
    iteration_log = []
    for iteration in range(max_iters + 1):
        check = initial_check if iteration == 0 and initial_check is not None else check_netlist(current_netlist)
        iteration_log.append({"iteration": iteration, "netlist": current_netlist,
                              "check": dict(check)})
        if (check["compile_success"] and check["dc_pass"]) or iteration == max_iters:
            return {**check, "netlist": current_netlist, "iterations": iteration,
                    "iteration_log": iteration_log}

        error_type = "compilation" if not check["compile_success"] else "DC simulation"
        # Preserve the exact simulator output as well as its extracted diagnostics.
        error_messages = "\n\n".join(
            f"{field}:\n{check[field]}"
            for field in ("compile_errors", "raw_output", "dc_errors", "dc_output")
            if check.get(field)
        ) or "Ngspice failed without diagnostic output."
        user_prompt = REPAIR_USER_TEMPLATE.format(
            error_type=error_type, netlist=current_netlist, error_messages=error_messages)
        if iteration:
            history = [{"iteration": item["iteration"], "check": item["check"]}
                       for item in iteration_log[:-1]]
            user_prompt += "\n\nEarlier simulator feedback for this sample:\n" + json.dumps(history, ensure_ascii=False)
        response = call_llm(system_prompt=REPAIR_SYSTEM_PROMPT, user_prompt=user_prompt,
                            image_path=image_path)
        new_netlist = parse_netlist_response(response)
        if not new_netlist:
            raise RuntimeError("Netlist repair API/format failure; rerun this sample to retry")
        current_netlist = new_netlist
        time.sleep(0.5)


def refine_description(netlist: str, original_description: str, max_attempts: int = 3) -> str:
    """Return a complete caption fitting CLIP's context, or a retryable error."""
    # Tokenization needs no CLIP weights/model download. Check the dependency
    # before paying for a generation request.
    from clip import tokenize
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    user_prompt = DESC_USER_TEMPLATE.format(
        netlist=netlist,
        original_description=original_description,
    )

    for attempt in range(max_attempts):
        response = call_llm(system_prompt=DESC_SYSTEM_PROMPT, user_prompt=user_prompt)
        refined = parse_description_response(response)
        if refined:
            try:
                tokenize([refined], context_length=77, truncate=False)
            except RuntimeError as error:
                if "too long" not in str(error).lower():
                    raise
                user_prompt += ("\nYour previous description exceeds the 77-token CLIP context "
                                "including start/end tokens. Shorten it without losing the "
                                "main circuit function and return <DESCRIPTION> only:\n" + refined)
            else:
                return refined
        else:
            user_prompt += "\nReturn a nonempty <DESCRIPTION>...</DESCRIPTION> response."
    raise ValueError(f"No valid description within 77 CLIP tokens after {max_attempts} attempts")


def _stage1_inputs(stage1_dir):
    """Require complete, versioned Stage 1 outputs before any paid request."""
    source = Path(stage1_dir)
    mapping = json.loads((source / "data_mapping.json").read_text(encoding="utf-8"))
    if not isinstance(mapping, list) or not mapping:
        raise ValueError("Stage 1 mapping must contain complete triplets")
    progress_path = source / "stage1_progress.json"
    if not progress_path.is_file():
        raise ValueError("Missing versioned Stage 1 progress; run the current Stage 1 into a new directory")
    progress = load_progress(progress_path, STAGE1_PROTOCOL_VERSION)
    if progress.get("failed"):
        raise ValueError("Stage 1 has retryable failures; complete Stage 1 before Stage 2")
    seen = {field: set() for field in ("image", "spice", "caption")}
    records = {}
    for entry in mapping:
        if not isinstance(entry, dict):
            raise ValueError("Invalid Stage 1 mapping entry")
        for field in seen:
            name = entry.get(field)
            if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError(f"Invalid Stage 1 {field} path: {name!r}")
            # Text outputs use their basename, so reject destination collisions too.
            identity = name if field == "image" else Path(name).name
            if identity in seen[field]:
                raise ValueError(f"Duplicate Stage 1 {field}: {name}")
            seen[field].add(identity)
            path = source / name
            if not path.is_file() or not path.stat().st_size:
                raise ValueError(f"Incomplete Stage 1 triplet: {name}")
            if field != "image" and not path.read_text(encoding="utf-8").strip():
                raise ValueError(f"Empty Stage 1 {field}: {name}")
        record = progress["completed"].get(entry["image"], {})
        if (record.get("image") != entry["image"]
                or record.get("output_spice") != entry["spice"]
                or record.get("output_caption") != entry["caption"]
                or record.get("source_mode") not in {"repair_existing", "generate_missing"}
                or not record.get("input_fingerprint")
                or not is_current(source, record, record["input_fingerprint"])):
            raise ValueError(f"Stage 1 output/provenance mismatch: {entry['image']}; rerun Stage 1")
        records[entry["image"]] = record
    if seen["image"] != set(progress["completed"]):
        raise ValueError("Stage 1 mapping/progress mismatch; rerun Stage 1 to finalize its outputs")
    return mapping, records


def run_stage2(data_dir, stage1_dir, output_dir, max_iters=5, start=0, end=None, workers=8, delay=0.2):
    """Run Stage 2 on the Stage 1 output; data_dir is checked only to keep outputs separate from inputs."""
    if not 0 <= max_iters <= MAX_REPAIR_ITERATIONS:
        raise ValueError(f"max_iters must be between 0 and {MAX_REPAIR_ITERATIONS}")
    if start < 0 or (end is not None and end < start) or workers < 1 or delay < 0:
        raise ValueError("Invalid Stage 2 range, workers or delay")
    source, output = Path(stage1_dir), Path(output_dir)
    if output.resolve() in {source.resolve(), Path(data_dir).resolve()}:
        raise ValueError("Stage 2 output must be separate from its inputs")
    mapping, upstream = _stage1_inputs(source)
    subset = mapping[start:end]
    progress_file = output / "stage2_progress.json"
    progress = load_progress(progress_file, PROTOCOL_VERSION)
    module_dir = Path(__file__).parent
    implementation = {name: module_dir / name for name in
                      ("stage2_refine.py", "pipeline_io.py", "spice_checker.py", "llm_client.py")}
    configuration = {
        "protocol_version": PROTOCOL_VERSION, "max_iters": max_iters,
        "model": llm_client.MODEL_NAME, "temperature": llm_client.TEMPERATURE,
        "max_tokens": llm_client.MAX_TOKENS, "api_url": llm_client.OPENROUTER_API_URL,
        "repair_system": REPAIR_SYSTEM_PROMPT, "repair_user": REPAIR_USER_TEMPLATE,
        "description_system": DESC_SYSTEM_PROMPT, "description_user": DESC_USER_TEMPLATE,
    }
    fingerprints = {}
    for entry in mapping:
        origin = upstream[entry["image"]]
        fingerprints[entry["image"]] = content_fingerprint(
            {**implementation, **{field: source / entry[field] for field in ("image", "spice", "caption")}},
            {**configuration, "entry": entry, "source_mode": origin["source_mode"],
             "stage1_input_fingerprint": origin["input_fingerprint"],
             "stage1_output_fingerprint": origin["output_fingerprint"]})
    for folder in ("spice", "captions", "images"):
        (output / folder).mkdir(parents=True, exist_ok=True)
    # Invalidate all stale entries, including entries outside a requested slice.
    for key, result in list(progress["completed"].items()):
        fingerprint = fingerprints.get(key)
        if fingerprint and result.get("input_fingerprint") == fingerprint:
            restore_image(source, output, result)
        if not fingerprint or not is_current(output, result, fingerprint):
            progress["completed"].pop(key)
            if fingerprint:
                progress["failed"][key] = {"success": False, "image": key,
                                            "error": "Inputs or outputs changed; rerun this sample"}
    progress["failed"] = {key: value for key, value in progress["failed"].items() if key in fingerprints}
    lock = threading.Lock()

    def process(entry):
        key = entry["image"]
        if key in progress["completed"]:
            return
        try:
            stage1_netlist = (source / entry["spice"]).read_text(encoding="utf-8")
            stage1_desc = (source / entry["caption"]).read_text(encoding="utf-8")
            initial_check = check_netlist(stage1_netlist)
            initially_verified = bool(initial_check["compile_success"] and initial_check["dc_pass"])
            if initially_verified:
                repair = {**initial_check, "netlist": stage1_netlist, "iterations": 0,
                          "iteration_log": [{"iteration": 0, "netlist": stage1_netlist,
                                             "check": dict(initial_check)}]}
            else:
                repair = repair_netlist(stage1_netlist, str(source / key), max_iters,
                                        initial_check=initial_check)
            verified = bool(repair["compile_success"] and repair["dc_pass"])
            refine = verified and not initially_verified
            description = refine_description(repair["netlist"], stage1_desc) if refine else stage1_desc
            if not description.strip() or not repair["netlist"].strip():
                raise ValueError(f"Empty Stage 2 output: {key}")
            origin = upstream[key]
            result = {"success": True, "image": key,
                      "compile_success": repair["compile_success"], "dc_pass": repair["dc_pass"],
                      "check": {field: value for field, value in repair.items()
                                if field not in {"netlist", "iterations", "iteration_log"}},
                      "initial_check": initial_check,
                      "action": "accepted" if initially_verified else ("repaired" if verified else "quality_failed"),
                      "repair_iterations": repair["iterations"],
                      "iteration_log": repair.get("iteration_log", []),
                      "source_mode": origin["source_mode"],
                      "original_spice": origin.get("original_spice", ""),
                      "original_caption": origin.get("original_caption", ""),
                      "stage1_input_fingerprint": origin["input_fingerprint"],
                      "description_refined": refine,
                      "input_fingerprint": fingerprints[key],
                      "output_spice": "spice/" + Path(entry["spice"]).name,
                      "output_caption": "captions/" + Path(entry["caption"]).name}
            (output / result["output_spice"]).write_text(repair["netlist"], encoding="utf-8")
            (output / result["output_caption"]).write_text(description, encoding="utf-8")
            copy_image(source, output, key)
            result["output_fingerprint"] = output_fingerprint(output, result)
        except Exception as error:
            result = {"success": False, "image": key, "error": str(error)}
        with lock:
            if output_ready(output, result):
                progress["completed"][key] = result
                progress["failed"].pop(key, None)
            else:
                progress["completed"].pop(key, None)
                progress["failed"][key] = result
            progress["summary"] = quality_summary(progress["completed"].values())
            atomic_json(progress_file, progress)
        if delay:
            time.sleep(delay)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(process, subset))
    successful = [progress["completed"][entry["image"]] for entry in mapping
                  if entry["image"] in progress["completed"]
                  and is_current(output, progress["completed"][entry["image"]], fingerprints[entry["image"]])]
    new_mapping = [{"image": result["image"], "spice": result["output_spice"],
                    "caption": result["output_caption"]} for result in successful]
    summary = quality_summary(successful)
    summary.update(failed=len(progress["failed"]), protocol_version=PROTOCOL_VERSION)
    progress["summary"] = summary
    atomic_json(progress_file, progress)
    atomic_json(output / "data_mapping.json", new_mapping)
    atomic_json(output / "stage2_summary.json", summary)
    print(f"Stage 2: {len(new_mapping)} complete triplets; {summary['failed']} retryable failures")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Netlist Repair & Description Refinement")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--stage1-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-iters", type=int, default=MAX_REPAIR_ITERATIONS)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delay", type=float, default=0.2)
    args = parser.parse_args()
    if shutil.which("ngspice") is None:
        parser.error("Ngspice must be installed before Stage 2 can verify or repair netlists")
    summary = run_stage2(args.data_dir, args.stage1_dir, args.output_dir, args.max_iters,
                         args.start, args.end, args.workers, args.delay)
    if summary["failed"]:
        raise SystemExit("Stage 2 has retryable failures; rerun to complete them before training")


if __name__ == "__main__":
    main()
