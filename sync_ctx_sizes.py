#!/usr/bin/env python3
import csv
import re
from pathlib import Path

ROOT = Path(__file__).parent
# Source of truth CSV (renamed from context-sizes.csv)
CSV_PATH = ROOT / "models.csv"
CONFIG_PATH = ROOT / "config.yaml"

HF_COL = "repo"
APPLIED_COL = "applied_ctx_size"

repo_ctx = {}
with CSV_PATH.open(newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        repo = row.get(HF_COL, "").strip()
        applied = row.get(APPLIED_COL, "").strip()
        if not repo or not applied or not applied.isdigit():
            continue
        repo_ctx[repo] = int(applied)

if not repo_ctx:
    print("No valid rows found in context-sizes.csv; nothing to do.")
    raise SystemExit(0)

content = CONFIG_PATH.read_text(encoding="utf-8")
lines = content.splitlines()

# Locate models: section start
models_start_idx = None
for idx, ln in enumerate(lines):
    if ln.strip() == "models:":
        models_start_idx = idx
        break

if models_start_idx is None:
    raise SystemExit("Could not find 'models:' section in config.yaml")

# Parse existing models entries: map repo -> (key, cmd_line)
repo_to_entry = {}
current_key = None
for ln in lines[models_start_idx+1:]:
    if not ln.startswith(" ") and ln.strip():
        # next top-level section (unlikely in current layout)
        break
    mkey = re.match(r"^\s+\"([^\"]+)\":\s*$", ln)
    if mkey:
        current_key = mkey.group(1)
        continue
    if current_key and "cmd:" in ln:
        # capture cmd line
        # Extract repo
        mrepo = re.search(r"-hf\s+([^\s:]+/[^\s:]+):[^\s]+", ln)
        if mrepo:
            repo = mrepo.group(1)
            repo_to_entry[repo] = (current_key, ln)
        current_key = None

if not repo_to_entry:
    raise SystemExit("No existing model entries found under models: in config.yaml")

# Helper to set/replace ctx-size in a cmd line
re_ctx = re.compile(r"--ctx-size\s+\d+")
def with_ctx(cmd_line: str, ctx: int) -> str:
    if re_ctx.search(cmd_line):
        return re_ctx.sub(f"--ctx-size {ctx}", cmd_line)
    m = re.search(r"(-hf\s+[^\s:]+/[^\s:]+:[^\s]+)", cmd_line)
    if m:
        end = m.end(1)
        return cmd_line[:end] + f" --ctx-size {ctx}" + cmd_line[end:]
    return cmd_line.rstrip() + f" --ctx-size {ctx}"

# Build new models section lines in CSV order
new_models_lines = ["models:"]
updated_count = 0
missing = []
for repo, ctx in repo_ctx.items():
    entry = repo_to_entry.get(repo)
    if not entry:
        missing.append(repo)
        continue
    key, old_cmd = entry
    new_cmd = with_ctx(old_cmd, ctx)
    new_models_lines.append(f"  \"{key}\":")
    new_models_lines.append(f"    {new_cmd.strip()}")
    updated_count += 1

head = lines[:models_start_idx]
new_content = "\n".join(head + new_models_lines) + "\n"
CONFIG_PATH.write_text(new_content, encoding="utf-8")

print(f"Rebuilt models section with {updated_count} entries from {CSV_PATH.name}.")
if missing:
    print("Warning: the following repos were not found in existing config and were skipped:")
    for r in missing:
        print(f"  - {r}")
