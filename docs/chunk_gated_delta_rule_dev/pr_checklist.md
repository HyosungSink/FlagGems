# PR Checklist for `chunk_gated_delta_rule`

## Title

```text
[FlagGems Operator Development Competition] Add chunk_gated_delta_rule operator
```

## Files

- [ ] Implementation file added in `src/flag_gems/ops/` or `src/flag_gems/fused/`.
- [ ] Export file updated.
- [ ] Registration updated if needed.
- [ ] Unit tests added.
- [ ] Benchmark added.
- [ ] No unrelated model repository or generated artifacts included.

## Correctness

- [ ] Reference implementation named and version recorded.
- [ ] Output matches reference.
- [ ] Final state matches reference if supported.
- [ ] Dtype coverage includes `float16` and/or `bfloat16`.
- [ ] Small and realistic shapes both pass.
- [ ] Optional `cu_seqlens` behavior tested if supported.

## Performance

- [ ] Benchmark compares against selected baseline.
- [ ] Benchmark includes short and long sequence lengths.
- [ ] Speedup table is included in PR body.

## Hygiene

- [ ] `pre-commit` or formatter passes.
- [ ] Focused diff.
- [ ] No vendored external project.
- [ ] Unsupported cases documented.
