#!/usr/bin/env python3
# Rebuilds the `models:` section in config.yaml from models.csv
# - models.csv columns: id, repo (owner/name:QUANT), hf_ctx_size, applied_ctx_size, type
#   * id is an arbitrary model identifier (e.g. name:QUANT) allowing multiple
#     entries referencing the same repo with differing context sizes.
#   * Lines starting with '#' are treated as comments and skipped.
#   * type column: +-separated modifiers (default: big GPU, fast SSD, normal inference)
#     - embedder: adds --embedding flag (instead of --ctx-size/--jinja)
#     - small: uses CUDA_VISIBLE_DEVICES=2 (instead of 0,1)
#     - slow: loads from /slow_models/ instead of /models/ (HDD storage)
#     Combine with +, e.g.: small+slow, embedder+slow
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

# Command templates: local (for .gguf files) and remote (for HF repos).
# Placeholders: {models_dir}, {id}, {repo}, {mode_flags}
CMD_LOCAL  = "${{llama-server}}\n  -m {models_dir}/{id}\n  {mode_flags}"
CMD_REMOTE = "${{llama-server}}\n  -hf {repo}\n  {mode_flags}"

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
            "modifiers": set(filter(None, (r.get("type") or "").strip().lower().split("+"))),
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

def build_cmd(model_id: str, repo_with_quant: str, ctx: int, modifiers: set) -> str:
    is_local = model_id.endswith(".gguf")
    mode_flags = "--embedding" if "embedder" in modifiers else f"--ctx-size {ctx}  --jinja"
    models_dir = "/slow_models" if "slow" in modifiers else "/models"
    template = CMD_LOCAL if is_local else CMD_REMOTE
    return template.format(models_dir=models_dir, id=model_id, repo=repo_with_quant, mode_flags=mode_flags)

def cuda_env(modifiers: set) -> str:
    if modifiers & {"embedder", "small"}:
        return "CUDA_VISIBLE_DEVICES=2"
    return "CUDA_VISIBLE_DEVICES=0,1"

# Validate + render the `groups:` section against the rebuilt model set.
# PyYAML can't emit comments (and safe_load strips them on read), so we render
# this block by hand as text. Dead/duplicate members are preserved as comments
# instead of being silently deleted, and fully-dead groups are commented out
# wholesale, so llama-swap never sees a member that isn't in `models:`.

def _scalar(value) -> str:
    # Render a YAML scalar the way the dumper would (lowercase true/false, etc.).
    # Take only the first line to drop the trailing '...' document-end marker that
    # safe_dump appends for bare scalars.
    return yaml.safe_dump(value, default_flow_style=True).splitlines()[0]

def render_groups(groups: dict, valid_keys: set):
    diagnostics = {"missing": [], "empty": [], "duplicate": []}
    seen = {}  # model_key -> first group that owns it
    lines = ["groups:"]

    for gname, gbody in groups.items():
        gbody = gbody if isinstance(gbody, dict) else {}
        raw_members = gbody.get("members")
        raw_members = raw_members if isinstance(raw_members, list) else []

        # Classify members, preserving original order.
        classified = []  # (member, kind, note)  kind in {active, missing, duplicate}
        has_active = False
        for m in raw_members:
            if m not in valid_keys:
                classified.append((m, "missing", "MISSING from models: (kept as comment)"))
                diagnostics["missing"].append((gname, m))
            elif m in seen:
                owner = seen[m]
                classified.append((m, "duplicate", f"DUPLICATE -- already in group '{owner}'"))
                diagnostics["duplicate"].append((gname, m, owner))
            else:
                seen[m] = gname
                classified.append((m, "active", None))
                has_active = True

        # Build this group's block lines (uncommented form).
        block = [f"  {gname}:"]
        for k, v in gbody.items():
            if k == "members":
                continue
            block.append(f"    {k}: {_scalar(v)}")
        block.append("    members:")
        for m, kind, note in classified:
            if kind == "active":
                block.append(f"    - {m}")
            else:
                block.append(f"    # - {m}   # {note}")
        if not classified:
            block.append("    # (no members)")

        # A group with no surviving active members is commented out wholesale.
        if not has_active:
            diagnostics["empty"].append(gname)
            block = [(f"# {ln}" if ln.strip() else "#") for ln in block]
            block.insert(0, f"  # GROUP '{gname}' EMPTY -- all members missing (kept as comment)")

        lines.extend(block)

    return "\n".join(lines) + "\n", diagnostics

# Rebuild models mapping in CSV order
new_models = {}
added_repos: List[str] = []
seen_keys = set()
for r in rows:
    repo_with_quant = r["repo"]
    ctx = r["applied"]
    model_id_raw = r["id"] or derive_key_from_repo(repo_with_quant)
    modifiers = r["modifiers"]

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
    cmd = build_cmd(model_id_raw, repo_with_quant, ctx, modifiers)

    preserved_entry = {}
    existing_entry = existing_models.get(yaml_key)
    if isinstance(existing_entry, dict):
        preserved_entry = dict(existing_entry)

    preserved_entry["cmd"] = cmd
    preserved_entry["env"] = [cuda_env(modifiers)]
    new_models[yaml_key] = preserved_entry
    added_repos.append(repo_with_quant)

# Write back config with the same top-level keys, replacing only 'models'
cfg["models"] = new_models
# Pull groups out so yaml.dump never emits it; we render it by hand (with comments)
# after validating its members against the rebuilt model set.
groups = cfg.pop("groups", None)
valid_keys = set(new_models.keys())
groups_text, group_diag = "", {"missing": [], "empty": [], "duplicate": []}
if isinstance(groups, dict) and groups:
    groups_text, group_diag = render_groups(groups, valid_keys)
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
    if groups_text:
        f.write(groups_text)

print(f"Rebuilt models section with {len(new_models)} entries from {CSV_PATH.name} using template.")

# Report groups validation results.
if isinstance(groups, dict) and groups:
    if not (group_diag["missing"] or group_diag["empty"] or group_diag["duplicate"]):
        print("groups: all members valid.")
    else:
        if group_diag["missing"]:
            print("groups: missing members (commented out, not in models:):")
            for gname, m in group_diag["missing"]:
                print(f"  - {gname}: {m}")
        if group_diag["duplicate"]:
            print("groups: duplicate members (commented out, kept in first group):")
            for gname, m, owner in group_diag["duplicate"]:
                print(f"  - {gname}: {m} (owned by '{owner}')")
        if group_diag["empty"]:
            print("groups: emptied groups (whole block commented out):")
            for gname in group_diag["empty"]:
                print(f"  - {gname}")
duplicate_warning = [k for k in new_models.keys() if "__" in k]
if duplicate_warning:
    print("Duplicate ids were detected; numeric suffixes applied:")
    for k in duplicate_warning:
        print(f"  - {k}")
print("Templates used:")
print(f"  local: {CMD_LOCAL}")
print(f"  remote: {CMD_REMOTE}")
