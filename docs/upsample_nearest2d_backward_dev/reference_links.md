# Reference Links for `upsample_nearest2d_backward`

## Competition

- ModelScope Track 1:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- FlagGems PR list:
  https://github.com/flagos-ai/FlagGems/pulls

## PyTorch

- Forward docs:
  https://pytorch.org/docs/stable/generated/torch.nn.functional.interpolate.html
- ATen schema in Python:

```python
torch.ops.aten.upsample_nearest2d.default._schema
torch.ops.aten.upsample_nearest2d_backward.default._schema
```

## Local Files

- `src/flag_gems/ops/upsample_nearest2d.py`
- `tests/test_upsample_nearest2d.py`
- `benchmark/test_upsample_nearest2d.py`
- `src/flag_gems/ops/upsample_bicubic2d_aa_backward.py`
- `tests/test_upsample_bicubic2d_aa.py`
- `benchmark/test_upsample_bicubic2d_aa.py`

## Public PRs to Watch

- https://github.com/flagos-ai/FlagGems/pull/2262
- https://github.com/flagos-ai/FlagGems/pull/1635
- https://github.com/flagos-ai/FlagGems/pull/1426
- https://github.com/flagos-ai/FlagGems/pull/2525

