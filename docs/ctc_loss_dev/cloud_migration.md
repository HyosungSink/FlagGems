# Cloud Migration Notes

This directory is ready to move to a cloud development environment.

## Recommended Upload

Move the whole repository directory:

```text
ctc_loss_flaggems_dev/
```

It contains the upstream FlagGems source, a dedicated branch, and the `ctc_loss`
development materials.

## First Commands on Cloud

```bash
cd ctc_loss_flaggems_dev
git status --short
bash scripts/ctc_loss/setup_cloud_env.sh
```

If the cloud environment provides a prebuilt FlagOS image, prefer its PyTorch,
Triton, and FlagTree packages over reinstalling everything locally.

## GPU Check

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

## Development Loop

```bash
# after implementing/registering ctc_loss
bash scripts/ctc_loss/run_accuracy_quick.sh
bash scripts/ctc_loss/run_accuracy_full.sh
bash scripts/ctc_loss/run_benchmark.sh
bash scripts/ctc_loss/check_pr_ready.sh
```

