# Cloud Migration Notes

This workspace is meant to be moved as a normal Git repository:

```text
upsample_nearest2d_backward_flaggems_dev/
```

## After Uploading

```bash
cd upsample_nearest2d_backward_flaggems_dev
git status
git remote -v
git branch --show-current
bash scripts/upsample_nearest2d_backward/setup_cloud_env.sh
```

If the cloud environment has GitHub access, refresh the base branch:

```bash
git fetch origin master
git rebase origin/master
```

If the cloud image already includes FlagGems dependencies, do not reinstall
everything. First run:

```bash
python scripts/upsample_nearest2d_backward/inspect_reference.py --device cuda
```

## Development Loop

```bash
python scripts/upsample_nearest2d_backward/inspect_reference.py --device cuda
bash scripts/upsample_nearest2d_backward/run_accuracy_quick.sh
bash scripts/upsample_nearest2d_backward/run_accuracy_full.sh
bash scripts/upsample_nearest2d_backward/run_benchmark.sh
```

## Expected Branch

```text
competition/upsample-nearest2d-backward-dev
```

