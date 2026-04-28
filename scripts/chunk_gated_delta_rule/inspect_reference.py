import importlib
import inspect


TARGETS = [
    (
        "FLA chunk module",
        "fla.ops.gated_delta_rule.chunk",
        ["chunk_gated_delta_rule", "ChunkGatedDeltaRuleFunction"],
    ),
    (
        "Megatron Core gated delta net",
        "megatron.core.ssm.gated_delta_net",
        ["torch_chunk_gated_delta_rule", "chunk_gated_delta_rule"],
    ),
    (
        "vLLM FLA ops",
        "vllm.model_executor.layers.fla.ops",
        ["fused_recurrent_gated_delta_rule"],
    ),
]


def main():
    for label, module_name, names in TARGETS:
        print(f"\n== {label}: {module_name}")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"unavailable: {type(exc).__name__}: {exc}")
            continue

        print("available")
        for name in names:
            obj = getattr(module, name, None)
            if obj is None:
                print(f"  {name}: missing")
                continue
            try:
                sig = inspect.signature(obj)
            except Exception:
                sig = "<signature unavailable>"
            print(f"  {name}: {sig}")


if __name__ == "__main__":
    main()
