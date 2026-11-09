"""Shared prompt loading and complete-triplet bookkeeping for refinement."""
import json
import hashlib
from pathlib import Path
import re
import shutil


def content_fingerprint(files, config):
    """Hash named input contents and the execution configuration, not mtimes."""
    contents = {}
    for label, value in sorted(files.items()):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing fingerprint input {label}: {path}")
        if not path.stat().st_size:
            raise ValueError(f"Empty fingerprint input {label}: {path}")
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
        contents[label] = digest.hexdigest()
    payload = json.dumps({'files': contents, 'config': config}, sort_keys=True,
                         ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def load_progress(path, version):
    """Refuse to reinterpret progress written by a different pipeline protocol."""
    path = Path(path)
    if not path.exists():
        return {'protocol_version': version, 'completed': {}, 'failed': {}}
    progress = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(progress, dict) or progress.get('protocol_version') != version:
        raise ValueError(f"Incompatible or legacy progress at {path}; use a new output directory "
                         f"for protocol {version}")
    if not all(isinstance(progress.get(key), dict) and
               all(isinstance(record, dict) for record in progress[key].values())
               for key in ('completed', 'failed')):
        raise ValueError(f"Invalid progress records at {path}; use a new output directory")
    return progress


def output_fingerprint(output_root, result):
    keys = ('image', 'output_spice', 'output_caption')
    if not all(isinstance(result.get(key), str) and result[key] for key in keys):
        raise ValueError('Output record must name image, output_spice and output_caption')
    return content_fingerprint({key: Path(output_root) / result[key] for key in keys}, {})


def is_current(output_root, result, input_fingerprint):
    try:
        if not output_ready(output_root, result) or result.get('input_fingerprint') != input_fingerprint:
            return False
        return result.get('output_fingerprint') == output_fingerprint(output_root, result)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load_prompts(filename):
    text = (Path(__file__).parent / "prompts" / filename).read_text(encoding="utf-8")
    prompts = []
    for heading in ("System Prompt", "User Prompt"):
        match = re.search(r"^## " + heading + r"\s*\n\n(`{3,})\n(.*?)\n\1\s*$",
                          text, re.MULTILINE | re.DOTALL)
        if match is None:
            raise ValueError(f"Missing executable {heading} block in {filename}")
        prompts.append(match.group(2))
    return tuple(prompts)


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def copy_image(source_root, output_root, image):
    source = Path(source_root) / image
    target = Path(output_root) / image
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)


def output_ready(output_root, result):
    if not result.get("success", False):
        return False
    for key in ("image", "output_spice", "output_caption"):
        if not result.get(key):
            return False
        path = Path(output_root) / result[key]
        if not path.is_file() or not path.stat().st_size:
            return False
        if key != "image" and not path.read_text(encoding="utf-8").strip():
            return False
    return True


def restore_image(source_root, output_root, result):
    """Restore an unchanged run's copied image without repeating paid requests."""
    if not result.get("success"):
        return
    text_paths = [Path(output_root) / result.get(key, "")
                  for key in ("output_spice", "output_caption")]
    if all(path.is_file() and path.read_text(encoding="utf-8").strip() for path in text_paths):
        image = result.get("image", "")
        if image and (Path(source_root) / image).is_file() and not (Path(output_root) / image).is_file():
            copy_image(source_root, output_root, image)


def quality_summary(records):
    checks = [record.get("check", record) for record in records]
    total = len(checks)
    compiled = sum(bool(check.get("compile_success")) for check in checks)
    dc = sum(bool(check.get("dc_pass")) for check in checks)
    both = sum(bool(check.get("compile_success") and check.get("dc_pass")) for check in checks)
    return {"total": total, "compile_pass": compiled, "dc_pass": dc, "both_pass": both,
            "compile_rate": round(100 * compiled / total, 2) if total else 0,
            "dc_rate": round(100 * dc / total, 2) if total else 0}
