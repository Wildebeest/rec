"""Compare prefix-cache rollback with cold prefill on the Flint checkpoint."""

import argparse
import json
from pathlib import Path

import mlx.core as mx

from rec_mlx.engine import Engine, Sampling


def equal(actual, expected):
    mx.eval(actual, expected)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert mx.array_equal(actual.view(mx.uint8), expected.view(mx.uint8)).item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Flint checkpoint directory")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="checkpoint manifest with the expected file hashes",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the results")
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=12305,
        help="prompt length to test (default: %(default)s)",
    )
    args = parser.parse_args()
    engine = Engine(args.model, args.manifest, context_limit=262144)
    ids = engine.prompt(
        [
            {
                "role": "user",
                "content": "Remember these notes. "
                + "The warehouse has seven blue crates. " * (args.prompt_tokens // 6),
            }
        ]
    )[: args.prompt_tokens]
    assert len(ids) == args.prompt_tokens
    settings = Sampling(temperature=0, top_p=1, top_k=0, max_tokens=16, seed=42)

    def generate(tokens, should_stop=lambda: False):
        events = list(engine.generate_tokens(tokens, settings, should_stop=should_stop))
        if not events:
            return None
        return events[-1]["metrics"]

    generate(ids)
    storage = engine._prompt_cache.storage
    boundary = max(
        item.position for item in engine._prompt_cache.checkpoints if item.position <= len(ids) - 7
    )
    changed = ids[:-7] + engine.tokenizer.encode(" new message", add_special_tokens=False)
    result = generate(changed)
    assert result["cached_prompt_tokens"] == boundary
    cache = engine._prompt_cache
    assert cache.storage is storage
    logits = mx.array(cache.logits)
    sliding = [
        tuple(mx.array(x) for x in layer._prior())
        for layer in cache.layers
        if hasattr(layer, "_prior")
    ]
    del cache, storage
    engine._prompt_cache = None
    # Match original chunk boundaries; a different prefill grouping can change BF16 rounding.
    generate(ids[:boundary])
    reference = generate(changed)
    equal(engine._prompt_cache.logits, logits)
    actual_sliding = [
        layer._prior() for layer in engine._prompt_cache.layers if hasattr(layer, "_prior")
    ]
    for actual, expected in zip(actual_sliding, sliding, strict=True):
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    assert result["token_ids"] == reference["token_ids"]
    engine._prompt_cache = None
    checks = 0

    def stop():
        nonlocal checks
        checks += 1
        return checks >= 2

    assert generate(ids, stop) is None
    completed = len(engine._prompt_cache.tokens)
    assert 0 < completed < len(ids)
    cancelled_storage = engine._prompt_cache.storage
    recovery_prompt = ids[:completed] + changed[completed:]
    recovery = generate(recovery_prompt)
    assert recovery["cached_prompt_tokens"] == completed
    assert engine._prompt_cache.storage is cancelled_storage
    del cancelled_storage
    recovered_logits = mx.array(engine._prompt_cache.logits)
    recovered_sliding = [
        tuple(mx.array(x) for x in layer._prior())
        for layer in engine._prompt_cache.layers
        if hasattr(layer, "_prior")
    ]
    engine._prompt_cache = None
    generate(ids[:completed])
    recovery_reference = generate(recovery_prompt)
    equal(engine._prompt_cache.logits, recovered_logits)
    reference_sliding = [
        layer._prior() for layer in engine._prompt_cache.layers if hasattr(layer, "_prior")
    ]
    for actual, expected in zip(reference_sliding, recovered_sliding, strict=True):
        for a, b in zip(actual, expected, strict=True):
            equal(a, b)
    assert recovery["token_ids"] == recovery_reference["token_ids"]
    report = {
        key: result[key]
        for key in (
            "prompt_tokens",
            "common_prefix_tokens",
            "cached_prompt_tokens",
            "prefilled_tokens",
            "prefill_seconds",
            "prefix_checkpoint_bytes",
        )
    }
    report.update(
        bitwise_logits_and_sliding=True,
        greedy_tokens_equal=True,
        cancelled_prefill_reused_tokens=completed,
        cancelled_prefill_bitwise_logits_and_sliding=True,
        cancelled_prefill_greedy_tokens_equal=True,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
