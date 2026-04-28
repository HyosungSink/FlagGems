# Cloud Migration Notes

This workspace is intended to be copied as a directory to a cloud GPU
development environment.

## After Upload

```bash
cd chunk_gated_delta_rule_flaggems_dev
git status
git remote -v
bash scripts/chunk_gated_delta_rule/setup_cloud_env.sh
```

## Recommended First Checks

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
PY
```

Then inspect the reference availability:

```bash
python scripts/chunk_gated_delta_rule/inspect_reference.py
```

## Development Loop

```bash
bash scripts/chunk_gated_delta_rule/run_accuracy_quick.sh
bash scripts/chunk_gated_delta_rule/run_accuracy_full.sh
bash scripts/chunk_gated_delta_rule/run_benchmark.sh
```

Before creating a PR:

```bash
bash scripts/chunk_gated_delta_rule/check_pr_ready.sh
```

## Syncing With Upstream

```bash
git fetch origin
git rebase origin/master
```

If conflicts appear, resolve them in the cloud environment and rerun accuracy
and benchmark checks.
