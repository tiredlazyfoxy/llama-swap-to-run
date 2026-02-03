#!/usr/bin/env python3
# Rebuilds the `models:` section in config.yaml from models.csv
# - models.csv columns: id, repo (owner/name:QUANT), hf_ctx_size, applied_ctx_size
#   * id is an arbitrary model identifier (e.g. name:QUANT) allowing multiple
#     entries referencing the same repo with differing context sizes.
#   * Lines starting with '#' are treated as comments and skipped.
#   * Lines with additional column "embedder" are treated as embedders.
# - Preserves other top-level sections in config.yaml
# - Uses existing per-repo flags like --flash-attn when present in current config

import csv
import sys
from pathlib import Path
from typing import List
import re

try:
    import yaml  # pip install pyyaml
except ImportError:
    print("Missing dependency: PyYAML. Install with: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

# Command template used for every model row.
# Available placeholders: {id} -> model id string, {repo} -> repo:quant string, {ctx} -> applied_ctx_size integer
# Adjust flags here to change generated commands globally.
# CMD_TEMPLATE = "llama-server --port ${{PORT}} -hf {repo} --ctx-size {ctx} --flash-attn --slots:${{SLOTS}}"
CMD_TEMPLATE = "${{llama-server}}\n  -m /models/{id}\n  --ctx-size {ctx}  --jinja"
CMD_TEMPLATE_REMOTE = "${{llama-server}}\n  -hf {repo}\n  --ctx-size {ctx}  --jinja"
CMD_EMBED = "${{llama-server}}\n  -m /models/{id}\n  --embedding"
CMD_EMBED_REMOTE = "${{llama-server}}\n -hf {repo}\n  --embedding"

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
            "embedder": (r.get("embedder") or "").strip().lower() == "embedder",
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
existing_models = models

# We no longer parse existing commands; regeneration is purely from CSV.

# Utility to derive a fallback key if needed (should be rare now that id exists)
def derive_key_from_repo(repo_with_quant: str) -> str:
    if ":" not in repo_with_quant or "/" not in repo_with_quant:
        return repo_with_quant.replace("/", "-")
    repo, quant = repo_with_quant.split(":", 1)
    name = repo.split("/")[-1].removesuffix("-GGUF")
    return f"{name}:{quant}"

# Utility to rebuild a cmd line, preserving extra flags from existing when present

def build_cmd(model_id: str, repo_with_quant: str, ctx: int, embedder: bool = False) -> str:
    if embedder:
        template = CMD_EMBED if model_id.endswith(".gguf") else CMD_EMBED_REMOTE
    else:
        template = CMD_TEMPLATE if model_id.endswith(".gguf") else CMD_TEMPLATE_REMOTE

    return template.format(id=model_id, repo=repo_with_quant, ctx=ctx)

# Rebuild models mapping in CSV order
new_models = {}
added_repos: List[str] = []
seen_keys = set()
for r in rows:
    repo_with_quant = r["repo"]
    ctx = r["applied"]
    model_id_raw = r["id"] or derive_key_from_repo(repo_with_quant)
    embedder = r["embedder"]

    # For multi-part models, derive a base name for the YAML key.
    # e.g. my-model-Q4-00001-of-00002.gguf -> my-model-Q4
    base_name = re.sub(r'-\d+-of-\d+(?=\.gguf|$)', '', model_id_raw, flags=re.IGNORECASE)
    yaml_key = base_name
    if yaml_key.lower().endswith('.gguf'):
        yaml_key = yaml_key[:-5]

    original_key = yaml_key
    suffix = 2
    while yaml_key in seen_keys:
        yaml_key = f"{original_key}__{suffix}"
        suffix += 1
    seen_keys.add(yaml_key)

    # The -m param must use the original filename from the CSV.
    cmd = build_cmd(model_id_raw, repo_with_quant, ctx, embedder)

    preserved_entry = {}
    existing_entry = existing_models.get(yaml_key)
    if isinstance(existing_entry, dict):
        preserved_entry = dict(existing_entry)

    preserved_entry["cmd"] = cmd
    new_models[yaml_key] = preserved_entry
    added_repos.append(repo_with_quant)

# Write back config with the same top-level keys, replacing only 'models'
cfg["models"] = new_models
# Custom YAML Dumper to ensure block style for 'cmd' keys
class BlockStyleDumper(yaml.SafeDumper):
    pass

def block_scalar_presenter(dumper, data):
    if isinstance(data, str):
        normalized = data.strip()
        if (
            "llama-server" in data
            or normalized.startswith("${{llama-server}}")
            or "\n" in data
            or len(data) > 80
        ):
            return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", data)

BlockStyleDumper.add_representer(str, block_scalar_presenter)

with CONFIG_PATH.open("w", encoding="utf-8") as f:
    yaml.dump(cfg, f, Dumper=BlockStyleDumper, sort_keys=False, allow_unicode=True)

print(f"Rebuilt models section with {len(new_models)} entries from {CSV_PATH.name} using template.")
duplicate_warning = [k for k in new_models.keys() if "__" in k]
if duplicate_warning:
    print("Duplicate ids were detected; numeric suffixes applied:")
    for k in duplicate_warning:
        print(f"  - {k}")
print("Templates used:")
print(f"  local: {CMD_TEMPLATE}")
print(f"  remote: {CMD_TEMPLATE_REMOTE}")
