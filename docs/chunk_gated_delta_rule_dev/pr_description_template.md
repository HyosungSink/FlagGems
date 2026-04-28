# [FlagGems Operator Development Competition] Add chunk_gated_delta_rule operator

## Summary

This PR adds `chunk_gated_delta_rule` support to FlagGems.

## Reference

Reference implementation:

```text
TODO: FLA / Megatron Core reference name and version
```

Supported signature:

```text
TODO: exact function signature
```

Supported cases:

- TODO

Unsupported cases:

- TODO

## Correctness

Commands:

```bash
pytest tests/test_chunk_gated_delta_rule.py -q
# or
pytest tests/test_FLA/test_chunk_gated_delta_rule.py -q
```

Results:

```text
TODO: paste summarized logs
```

## Performance

Command:

```bash
pytest benchmark/test_FLA/test_chunk_gated_delta_rule_perf.py -s --record log
```

Results:

```text
TODO: paste benchmark table
```

## Checklist

- [ ] Correctness tests pass.
- [ ] Benchmark passes.
- [ ] Code is formatted.
- [ ] Diff is scoped to `chunk_gated_delta_rule`.
