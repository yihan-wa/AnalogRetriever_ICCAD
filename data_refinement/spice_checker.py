"""
SPICE Netlist Compilation & DC Simulation Checker.
Tests netlists for:
  1) Ngspice compilation success
  2) DC operating point (.op) simulation pass
"""

import subprocess
import tempfile
import os
import json
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def _device_line(line):
    """Recognize a component card, including common named sources such as VIN."""
    parts = line.split(';', 1)[0].strip().split()
    if not parts or len(parts[0]) < 2:
        return False
    kind = parts[0][0].upper()
    if kind not in "RCLVIDMQEFGHBKSTWXJZUYONPA":
        return False
    # A prose title such as "Voltage divider circuit example" is not a card.
    # Keep recognizable malformed cards too: ngspice must report their error,
    # rather than accidentally consuming e.g. "V1 1 0" as the title.
    return (any(char.isdigit() for char in parts[0])
            or any(re.match(r'^[+\-]?(?:\d|\.\d|\{)', token) for token in parts[1:])
            or any(token.upper().startswith(('DC', 'AC', 'PULSE(', 'SIN(', 'PWL('))
                   for token in parts[3:]))


def preprocess_netlist(netlist_text: str) -> str:
    """Reject empty circuits and protect a device-first deck with a title."""
    lines = netlist_text.strip().splitlines()
    for index, line in enumerate(lines):
        if re.match(r'^\.end(?:\s|$)', line.strip(), re.I):
            lines = lines[:index + 1]
            break
    if not lines or not any(_device_line(line) for line in lines):
        raise ValueError("Empty netlist: no component cards found")
    first_line = lines[0].strip()
    if first_line.startswith('.') or _device_line(first_line):
        lines.insert(0, "* SPICE Netlist")
    if not any(re.match(r'^\.end(?:\s|$)', line.strip(), re.I) for line in lines):
        lines.append('.end')
    return '\n'.join(lines) + '\n'


def dc_only_netlist(netlist_text: str) -> str:
    """Run only .op: unrelated transient/control commands cannot decide DC pass."""
    lines = preprocess_netlist(netlist_text).splitlines()
    output = [lines[0]]
    in_control = False
    skip_continuation = False
    analysis = {'.op', '.ac', '.dc', '.tran', '.noise', '.pz', '.tf', '.disto',
                '.sens', '.sp', '.print', '.plot', '.four', '.meas', '.measure', '.save'}
    for line in lines[1:]:
        stripped = line.split(';', 1)[0].strip()
        directive = stripped.split()[0].lower() if stripped else ''
        if directive == '.control':
            in_control = True
            continue
        if directive == '.endc':
            in_control = False
            continue
        if in_control:
            continue
        if stripped.startswith('+') and skip_continuation:
            continue
        skip_continuation = directive in analysis
        if skip_continuation:
            continue
        if directive == '.end':
            output.extend(['.op', '.end'])
            break
        output.append(line)
    return '\n'.join(output) + '\n'


def check_compilation(netlist_text: str, timeout: int = 15) -> dict:
    """
    Check if a netlist compiles (parses + loads) in ngspice.
    Uses batch mode which also attempts simulation — we detect both
    parse-level errors and model/component-level errors.
    Returns dict with 'success' bool and 'errors' string.
    """
    result = {"compile_success": False, "compile_errors": "", "raw_output": ""}

    try:
        # Adding a known analysis avoids ngspice's ambiguous "empty netlist or
        # no .plot/.print" error for a valid circuit with no analysis command.
        processed = dc_only_netlist(netlist_text)
    except ValueError as error:
        result["compile_errors"] = str(error)
        return result

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
        f.write(processed)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ['ngspice', '-b', tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        output = proc.stdout + proc.stderr
        result["raw_output"] = output

        # Compile-level errors ONLY: syntax, missing models, bad component lines
        # Excludes DC-simulation errors (singular matrix, convergence) —
        # those are handled separately in check_dc_simulation().
        compile_error_patterns = [
            r'(?i)mal\s*formed',
            r'(?i)bad syntax',
            r'(?i)unrecognized',
            r'(?i)could not find a valid modelname',
            r'(?i)can\'t find model',
            r'(?i)unknown subckt',
            r'(?i)undefined reference',
            r'(?i)missing.*model',
            r'(?i)no .end',
            r'(?i)fatal error',
            r'(?i)not a valid',
            r'(?i)unknown device',
            r'(?i)unresolved',
            r'(?i)invalid.*parameter',
            r'(?i)can\'t open',
            r'(?i)parser error',
            r'(?i)not enough parameters',
            r'(?i)incomplete or empty netlist',
        ]

        # DC-simulation errors to EXCLUDE from compile check
        dc_only_patterns = [
            r'(?i)singular matrix',
            r'(?i)no convergence',
            r'(?i)timestep too small',
            r'(?i)doAnalyses.*Too many iterations',
            r'(?i)Trying gmin',
        ]

        has_error = False
        error_lines = []
        for line in output.split('\n'):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Skip lines that are DC-simulation errors, not compile errors
            is_dc_error = any(re.search(p, line_stripped) for p in dc_only_patterns)
            if is_dc_error:
                continue
            for pat in compile_error_patterns:
                if re.search(pat, line_stripped):
                    has_error = True
                    error_lines.append(line_stripped)
                    break

        # Also check exit code combined with "Error on line" pattern
        error_on_line = re.findall(r'Error on line \d+', output)
        if error_on_line:
            has_error = True
            error_lines.extend(error_on_line)

        # Known operating-point failures do not imply a parse error, but a
        # simulator crash or unexplained nonzero exit cannot count as success.
        known_dc_failure = any(re.search(pattern, output) for pattern in dc_only_patterns)
        if proc.returncode < 0 or (proc.returncode != 0 and not known_dc_failure):
            has_error = True
            error_lines.append(f"Ngspice exited with status {proc.returncode}")

        if has_error:
            result["compile_success"] = False
            result["compile_errors"] = '\n'.join(error_lines[:20])
        else:
            result["compile_success"] = True

    except subprocess.TimeoutExpired:
        result["compile_errors"] = "Timeout during compilation"
    except Exception as e:
        result["compile_errors"] = str(e)
    finally:
        os.unlink(tmp_path)

    return result


def check_dc_simulation(netlist_text: str, timeout: int = 15) -> dict:
    """
    Check if a netlist passes DC operating point simulation.
    Returns dict with 'dc_pass' bool and 'dc_errors' string.
    """
    result = {"dc_pass": False, "dc_errors": "", "dc_output": ""}

    try:
        processed = dc_only_netlist(netlist_text)
    except ValueError as error:
        result["dc_errors"] = str(error)
        return result

    with tempfile.NamedTemporaryFile(mode='w', suffix='.cir', delete=False) as f:
        f.write(processed)
        tmp_path = f.name

    try:
        proc = subprocess.run(
            ['ngspice', '-b', tmp_path],
            capture_output=True, text=True, timeout=timeout
        )
        output = proc.stdout + proc.stderr
        result["dc_output"] = output

        # Check for simulation errors using regex for precision
        error_patterns_dc = [
            r'(?i)^Error',                         # Lines starting with Error
            r'(?i)^ERROR',                         # Lines starting with ERROR
            r'(?i)fatal error',
            r'(?i)no convergence',
            r'(?i)timestep too small',
            r'(?i)doAnalyses.*Too many iterations',
            r'(?i)Simulation interrupted due to error',
            r'(?i)could not find a valid modelname',
            r'(?i)can\'t find model',
            r'(?i)not enough parameters',
        ]

        has_error = False
        error_lines = []
        for line in output.split('\n'):
            line_stripped = line.strip()
            if not line_stripped:
                continue
            # Skip informational lines (title echo, memory info, etc.)
            if line_stripped.startswith('Circuit:') or line_stripped.startswith('No compatibility'):
                continue
            if 'note:' in line_stripped.lower():
                continue
            for pat in error_patterns_dc:
                if re.search(pat, line_stripped):
                    has_error = True
                    error_lines.append(line_stripped)
                    break

        has_operating_point = re.search(r'No\. of Data Rows\s*:\s*[1-9]\d*', output) is not None
        if proc.returncode != 0 or has_error or not has_operating_point:
            result["dc_pass"] = False
            result["dc_errors"] = '\n'.join(error_lines[:20]) or "No successful operating-point result"
        else:
            result["dc_pass"] = True

    except subprocess.TimeoutExpired:
        result["dc_errors"] = "Timeout during DC simulation"
    except Exception as e:
        result["dc_errors"] = str(e)
    finally:
        os.unlink(tmp_path)

    return result


def check_netlist(netlist_text: str) -> dict:
    """Full check: compilation + DC simulation."""
    compile_result = check_compilation(netlist_text)
    dc_result = {"dc_pass": False, "dc_errors": "Skipped (compilation failed)"}

    if compile_result["compile_success"]:
        dc_result = check_dc_simulation(netlist_text)

    return {
        **compile_result,
        **dc_result,
    }


def check_dataset(data_dir: str, mapping_file: str = None, max_workers: int = 8) -> dict:
    """
    Check all netlists in a dataset directory.
    Returns summary statistics and per-file results.
    """
    if mapping_file:
        with open(mapping_file) as f:
            mapping = json.load(f)
        spice_files = []
        for entry in mapping:
            spice_path = os.path.join(data_dir, entry["spice"])
            spice_files.append((entry, spice_path))
    else:
        spice_dir = os.path.join(data_dir, "spice")
        spice_files = []
        for fn in sorted(os.listdir(spice_dir)):
            if fn.endswith('.txt') or fn.endswith('.cir') or fn.endswith('.sp'):
                spice_files.append(({"spice": fn}, os.path.join(spice_dir, fn)))

    total = len(spice_files)
    results = []

    def _check_one(item):
        entry, path = item
        try:
            with open(path) as f:
                netlist = f.read()
            check = check_netlist(netlist)
            return {
                "file": entry.get("spice", path),
                "image": entry.get("image", ""),
                **check,
            }
        except Exception as e:
            return {
                "file": entry.get("spice", path),
                "image": entry.get("image", ""),
                "compile_success": False,
                "compile_errors": str(e),
                "dc_pass": False,
                "dc_errors": str(e),
            }

    print(f"Checking {total} netlists...")
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_check_one, item): item for item in spice_files}
        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            completed += 1
            if completed % 100 == 0:
                print(f"  Progress: {completed}/{total}")

    compile_pass = sum(1 for r in results if r["compile_success"])
    dc_pass = sum(1 for r in results if r["dc_pass"])
    both_pass = sum(1 for r in results if r["compile_success"] and r["dc_pass"])

    summary = {
        "total": total,
        "compile_pass": compile_pass,
        "compile_rate": round(compile_pass / total * 100, 2) if total > 0 else 0,
        "dc_pass": dc_pass,
        "both_pass": both_pass,
        "dc_rate": round(dc_pass / total * 100, 2) if total > 0 else 0,
    }

    print(f"\n=== Dataset Quality Summary ===")
    print(f"Total samples:    {summary['total']}")
    print(f"Compile pass:     {summary['compile_pass']} ({summary['compile_rate']}%)")
    print(f"DC sim pass:      {summary['dc_pass']} ({summary['dc_rate']}%)")

    return {"summary": summary, "details": results}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Check SPICE netlist quality")
    parser.add_argument("--data-dir", required=True, help="Dataset directory")
    parser.add_argument("--mapping", default=None, help="data_mapping.json path")
    parser.add_argument("--output", default=None, help="Output JSON path")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    results = check_dataset(args.data_dir, args.mapping, args.workers)

    if args.output:
        # Save only summary + file-level pass/fail (not raw output)
        save_data = {
            "summary": results["summary"],
            "details": [
                {
                    "file": r["file"],
                    "compile_success": r["compile_success"],
                    "compile_errors": r.get("compile_errors", ""),
                    "dc_pass": r["dc_pass"],
                    "dc_errors": r.get("dc_errors", ""),
                }
                for r in results["details"]
            ]
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Results saved to {args.output}")
