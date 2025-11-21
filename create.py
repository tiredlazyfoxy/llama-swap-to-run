#!/usr/bin/env python3
# Rebuilds the `models:` section in config.yaml from models.csv
# - models.csv columns: id, repo (owner/name:QUANT), hf_ctx_size, applied_ctx_size
#   * id is an arbitrary model identifier (e.g. name:QUANT) allowing multiple
#     entries referencing the same repo with differing context sizes.
#   * Lines starting with '#' are treated as comments and skipped.
# - Preserves other top-level sections in config.yaml
# - Uses existing per-repo flags like --flash-attn when present in current config

import csv
import sys
from pathlib import Path
from typing import Dict, Tuple

try:
    import yaml  # pip install pyyaml
except ImportError:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).parent
CSV_PATH = ROOT / "models.csv"
CONFIG_PATH = ROOT / "config.yaml"

def load_csv_rows(path: Path):
    raw = path.read_text(encoding="utf-8").splitlines()
    filtered = [ln for ln in raw if not ln.lstrip().startswith('#') and ln.strip()]
    if not filtered:
        return []
    reader = csv.DictReader(filtered)
    required = {"repo", "applied_ctx_size"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        print(f"CSV missing columns: {missing}", file=sys.stderr)
        sys.exit(2)
    rows_local = []
    for r in reader:
        repo = (r.get("repo") or "").strip()
        if not repo:
            continue
        id_raw = (r.get("id") or "").strip()
        # Fallback: derive id if absent
        if not id_raw:
            id_raw = repo.split('/')[-1].split(':')[0]
            if ':' in repo:
                id_raw += ':' + repo.split(':', 1)[1]
        rows_local.append({
            "id": id_raw,
            "repo": repo,
            "applied": int((r.get("applied_ctx_size") or "0").strip() or 0),
            "hf": (r.get("hf_ctx_size") or "").strip(),
        })
    return rows_local

rows = load_csv_rows(CSV_PATH)

if not rows:
    print("No rows found in models.csv; nothing to do.", file=sys.stderr)
    sys.exit(0)

# Load current config
with CONFIG_PATH.open("r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

if not isinstance(cfg, dict):
    print("config.yaml is not a mapping.", file=sys.stderr)
    sys.exit(3)

models = cfg.get("models")
if models is None or not isinstance(models, dict):
    print("config.yaml has no 'models' mapping.", file=sys.stderr)
    sys.exit(4)

# Build helper map from existing models: repo_with_quant -> (key, cmd_str)
repo_to_existing: Dict[str, Tuple[str, str]] = {}
for key, val in models.items():
    if not isinstance(val, dict):
        continue
    cmd = val.get("cmd")
    if not isinstance(cmd, str):
        continue
    # Find " -hf <owner>/<name>:<quant> " token
    parts = cmd.split()
    for i, p in enumerate(parts):
        if p == "-hf" and i + 1 < len(parts):
            repo_with_quant = parts[i + 1]
            repo_to_existing[repo_with_quant] = (key, cmd)
            break

# Utility to derive a fallback key if needed (should be rare now that id exists)
def derive_key_from_repo(repo_with_quant: str) -> str:
    if ":" not in repo_with_quant or "/" not in repo_with_quant:
        return repo_with_quant.replace("/", "-")
    repo, quant = repo_with_quant.split(":", 1)
    name = repo.split("/")[-1].removesuffix("-GGUF")
    return f"{name}:{quant}"

# Utility to rebuild a cmd line, preserving extra flags from existing when present

def build_cmd(existing_cmd: str, repo_with_quant: str, ctx: int) -> str:
    base = f"llama-server --port ${{PORT}} -hf {repo_with_quant} --ctx-size {ctx}"
    if not existing_cmd:
        # Default: append --flash-attn to match previous usage
        return base + " --flash-attn"
    # Preserve flags that were present in existing_cmd but not in base
    # We keep it simple: if '--flash-attn' was present, add it; otherwise leave as-is
    extra = []
    if "--flash-attn" in existing_cmd and "--flash-attn" not in base:
        extra.append("--flash-attn")
    if extra:
        return base + " " + " ".join(extra)
    return base

# Rebuild models mapping in CSV order
new_models = {}
missing_in_cfg = []
seen_keys = set()
for r in rows:
    repo_with_quant = r["repo"]
    ctx = r["applied"]
    model_id = r["id"] or derive_key_from_repo(repo_with_quant)
    # Ensure uniqueness; if duplicate id encountered, append numeric suffix
    original_id = model_id
    suffix = 2
    while model_id in seen_keys:
        model_id = f"{original_id}__{suffix}"
        suffix += 1
    seen_keys.add(model_id)
    existing_key, existing_cmd = repo_to_existing.get(repo_with_quant, (None, None))
    cmd = build_cmd(existing_cmd or "", repo_with_quant, ctx)
    new_models[model_id] = {"cmd": cmd}
    if existing_key is None and repo_with_quant not in repo_to_existing:
        missing_in_cfg.append(repo_with_quant)

# Write back config with the same top-level keys, replacing only 'models'
cfg["models"] = new_models
with CONFIG_PATH.open("w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

print(f"Rebuilt models section with {len(new_models)} entries from {CSV_PATH.name}.")
duplicate_warning = [k for k in new_models.keys() if "__" in k]
if duplicate_warning:
    print("Duplicate ids were detected; numeric suffixes applied:")
    for k in duplicate_warning:
        print(f"  - {k}")
if missing_in_cfg:
    print("Note: the following repo:quant entries were not in the original config and were added:")
    for x in missing_in_cfg:
        print(f"  - {x}")
