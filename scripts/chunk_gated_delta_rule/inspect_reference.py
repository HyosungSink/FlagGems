import importlib
import importlib.metadata as metadata
import inspect
import ast


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

SOURCE_FALLBACKS = {
    "fla.ops.gated_delta_rule.chunk": (
        "fla-core",
        "fla/ops/gated_delta_rule/chunk.py",
    ),
    "megatron.core.ssm.gated_delta_net": (
        "megatron-core",
        "megatron/core/ssm/gated_delta_net.py",
    ),
}


def _inspect_source(module_name, names):
    fallback = SOURCE_FALLBACKS.get(module_name)
    if fallback is None:
        return

    dist_name, rel_path = fallback
    try:
        dist = metadata.distribution(dist_name)
    except metadata.PackageNotFoundError:
        print(f"  source fallback: {dist_name} distribution missing")
        return

    path = dist.locate_file(rel_path)
    if not path.exists():
        print(f"  source fallback: {dist_name} {metadata.version(dist_name)} missing {rel_path}")
        return

    print(f"  source fallback: {dist_name} {metadata.version(dist_name)} at {path}")
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"  source parse failed: {type(exc).__name__}: {exc}")
        return

    definitions = {
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    for name in names:
        print(f"  {name}: {'defined in source' if name in definitions else 'missing in source'}")


def main():
    for label, module_name, names in TARGETS:
        print(f"\n== {label}: {module_name}")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            print(f"unavailable: {type(exc).__name__}: {exc}")
            _inspect_source(module_name, names)
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
