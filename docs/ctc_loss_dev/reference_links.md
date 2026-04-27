# Reference Links

## Official / Primary

- Competition statement:
  https://www.modelscope.cn/events/180/%E3%80%90Track%201%20-%E7%AE%97%E5%AD%90%E5%BC%80%E5%8F%91%E5%92%8C%E6%80%A7%E8%83%BD%E6%8C%91%E6%88%98%E3%80%91%E8%B5%9B%E9%A2%98%E8%AF%B4%E6%98%8E
- FlagGems repository:
  https://github.com/flagos-ai/FlagGems
- PyTorch `ctc_loss` API:
  https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.ctc_loss.html
- PyTorch source search:
  https://github.com/pytorch/pytorch/search?q=ctc_loss&type=code

## Relevant Public PRs

Use these only for competitive analysis and test/benchmark ideas. Do not assume
they satisfy the competition gate.

- ctc_loss implementation:
  https://github.com/flagos-ai/FlagGems/pull/2103
- ctc_loss implementation:
  https://github.com/flagos-ai/FlagGems/pull/1775
- ctc_loss registration:
  https://github.com/flagos-ai/FlagGems/pull/1967

## Local Repository Files to Study

- `src/flag_gems/ops/mse_loss.py`
- `src/flag_gems/ops/nllloss.py`
- `src/flag_gems/ops/nll_loss_nd.py`
- `tests/test_cross_entropy_loss.py`
- `tests/test_nll_loss_forward.py`
- `benchmark/test_cross_entropy_loss.py`
- `benchmark/test_nll_loss.py`

