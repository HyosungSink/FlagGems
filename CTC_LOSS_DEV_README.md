# CTC Loss Competition Workspace

This repository is a FlagGems development workspace prepared for the
`ctc_loss` operator task in the FlagOS / ModelScope Track 1 competition.

Primary task:

```text
torch.nn.functional.ctc_loss(
    log_probs,
    targets,
    input_lengths,
    target_lengths,
    blank=0,
    reduction="mean",
    zero_infinity=False,
)
```

The goal is not only to pass CI. The competition asks for correctness,
performance, open-source integration quality, compatibility, complete tests,
and readable code. A PR is only safe when all of those are defensible.

## Where to Start

Read these files first:

- `docs/ctc_loss_dev/competition_requirements.md`
- `docs/ctc_loss_dev/implementation_plan.md`
- `docs/ctc_loss_dev/test_matrix.md`
- `docs/ctc_loss_dev/benchmark_plan.md`
- `docs/ctc_loss_dev/pr_checklist.md`

Then run the environment check:

```bash
bash scripts/ctc_loss/setup_cloud_env.sh
```

When the operator is implemented and registered, use:

```bash
bash scripts/ctc_loss/run_accuracy_quick.sh
bash scripts/ctc_loss/run_accuracy_full.sh
bash scripts/ctc_loss/run_benchmark.sh
bash scripts/ctc_loss/check_pr_ready.sh
```

## Expected Code Touch Points

The implementation will likely touch:

- `src/flag_gems/ops/ctc_loss.py`
- `src/flag_gems/ops/__init__.py`
- `src/flag_gems/__init__.py`
- `tests/test_ctc_loss.py`
- `benchmark/test_ctc_loss.py`
- `benchmark/core_shapes.yaml`

Do not register a placeholder operator. Register only after the forward path,
backward path, tests, and benchmark are coherent enough for local validation.

## Important Risk

The known hard part is not merely implementing dynamic programming. The hard
part is keeping the slowest benchmark cases above the competition threshold.
The public PRs that remain unmerged suggest that `ctc_loss` backward can fall
below `0.9x` in some dtypes or shapes. Treat worst-case benchmark speed as a
first-class requirement from the beginning.

