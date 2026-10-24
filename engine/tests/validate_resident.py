"""Compare resident decoding with scalar and native MLX decoding on the Flint checkpoint."""

import argparse
import gc
import hashlib
import json
import statistics
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm.sample_utils import make_sampler
from transformers import AutoTokenizer

from rec_mlx.engine import Engine, Sampling
from rec_mlx.kernels import build_decoder, megakernel
from rec_mlx.kernels.masked_sampling import FilteredResidentSampler
from rec_mlx.quantized import QATModel


def validate_stopping(engine, resident, specification, tokenizer, prompt):
    model = engine.model
    original_eos, original_filtered = model.eos_tokens, resident.filtered
    eos_cases, cancellation_cases = [], []
    modes = ((0, 1, 0), (1, 1, 0), (0.6, 0.95, 20))
    try:
        for temperature, top_p, top_k in modes:
            settings = dict(temperature=temperature, top_p=top_p, top_k=top_k)
            sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)

            def pick(logits, index):
                flat = logits.reshape(-1).astype(mx.float32)
                if temperature == 0:
                    return int(mx.argmax(flat).item())
                if (temperature, top_p, top_k) == (1, 1, 0):
                    return int(mx.random.categorical(flat, key=mx.random.key(42 + index)).item())
                return int(sample(flat - mx.logsumexp(flat)).item())

            discovery = engine.make_session(max_new_tokens=33)
            logits = discovery.prefill(prompt)
            mx.random.seed(42)
            tokens, keys = [pick(logits, 0)], [mx.random.state[0]]
            for index in range(32):
                tokens.append(pick(discovery.forward_token(tokens[-1]), index + 1))
                keys.append(mx.random.state[0])
            del discovery
            starts = [
                begin
                for begin in range(25)
                if all(tokens[begin + stop] not in tokens[: begin + stop] for stop in (1, 4, 8))
            ]
            assert starts, "Prompt did not produce unique early/middle/final EOS fixtures"
            begin = starts[0]
            for stop in (1, 4, 8):
                reference = engine.make_session(max_new_tokens=33)
                reference.prefill(prompt)
                for token in tokens[:begin]:
                    reference.forward_token(token)
                mx.random.state[0] = keys[begin]
                expected = [
                    pick(reference.forward_token(tokens[begin + i]), begin + i + 1)
                    for i in range(stop)
                ]
                expected_key = mx.random.state[0]
                ref_pos = reference._state["position"]
                ref_kv = kv_digest(reference._state["kv"], ref_pos)
                del reference
                gc.collect()
                mx.synchronize()
                mx.clear_cache()

                candidate = engine.make_session(max_new_tokens=33)
                candidate.prefill(prompt)
                for token in tokens[:begin]:
                    candidate.forward_token(token)
                model.eos_tokens = [expected[-1]]
                # Each compiled graph must capture this fixture's EOS, not a previous one.
                resident.filtered = FilteredResidentSampler(resident.module, engine, specification)
                mx.random.state[0] = keys[begin]
                actual = resident.block(candidate._state, tokens[begin], 8, 43 + begin, **settings)
                assert actual == expected and len(actual) == stop, "EOS truncation mismatch"
                assert mx.array_equal(mx.random.state[0], expected_key).item(), "EOS RNG mismatch"
                assert candidate._state["position"] == ref_pos
                assert kv_digest(candidate._state["kv"], candidate._state["position"]) == ref_kv, (
                    "EOS wrote beyond the retained prefix"
                )
                eos_cases.append(dict(**settings, stop=stop, packet_start=begin))
                del candidate
                gc.collect()
                mx.synchronize()
                mx.clear_cache()
    finally:
        model.eos_tokens, resident.filtered = original_eos, original_filtered

    api = Engine.__new__(Engine)
    api._prefix_preparation = None
    api.mx, api.runtime, api.resident_sampler = mx, engine, resident
    api.tokenizer, api.context_limit, api.eos = tokenizer, model.context_limit, set(original_eos)
    mx.eval(model.native.parameters())
    for temperature, top_p, top_k in modes:
        api._prompt_cache = None
        sampling = Sampling(
            temperature=temperature, top_p=top_p, top_k=top_k, max_tokens=17, seed=42
        )
        emitted = []
        for event in api.generate_tokens(prompt, sampling, should_stop=lambda: len(emitted) == 2):
            if "token" in event:
                emitted.append(event["token"])
            else:
                cancelled = event
        assert cancelled["finish_reason"] == "stop" and len(emitted) == 2
        assert cancelled["metrics"]["computed_tokens"] == 9, "Cancellation did not discard a packet"
        cached = list(api.generate_tokens(prompt, sampling))[-1]["metrics"]
        api._prompt_cache = None
        cold = list(api.generate_tokens(prompt, sampling))[-1]["metrics"]
        assert cached["cached_prompt_tokens"] == len(prompt) and cold["cached_prompt_tokens"] == 0
        assert cached["token_ids"] == cold["token_ids"], (
            "Cancelled packet contaminated prefix reuse"
        )
        cancellation_cases.append(dict(temperature=temperature, top_p=top_p, top_k=top_k))
    api._prompt_cache = None
    mx.clear_cache()
    return dict(eos=eos_cases, cancellation=cancellation_cases)


def same_kv(a, b):
    return all(mx.array_equal(x, y).item() for x, y in zip(a.banks, b.banks, strict=True))


def kv_digest(kv, pos):
    h = hashlib.blake2b(digest_size=32)
    h.update(pos.to_bytes(8, "little"))
    bank0 = kv.banks[0].view(mx.uint16)
    tail_maxes = [mx.max(bank.view(mx.uint16)) for bank in kv.banks[1:]]
    active_slices = []
    for row in kv.rows[:49]:
        if row[1]:
            offset, cap = row[61:63]
            raw = bank0[offset // 1024 : offset // 1024 + cap].reshape(8, cap, 128)
            for head in range(8):
                if pos < cap:
                    tail_maxes.append(mx.max(raw[head, pos:]))
                active_slices.append(raw[head, :pos])
    local_slice = bank0[kv.global_elements // 1024 :]
    tail_max = mx.max(mx.stack(tail_maxes))
    mx.eval(tail_max, *active_slices, local_slice)
    assert tail_max.item() == 0, "KV wrote beyond position"
    for s in active_slices:
        h.update(np.array(s).tobytes())
    h.update(np.array(local_slice).tobytes())
    return h.digest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Flint checkpoint directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON file for the results")
    args = parser.parse_args()
    mx.set_cache_limit(512 * 1024**2)
    model = QATModel(args.model)
    specification = {}
    metal = megakernel._metal

    def capture_specification(name, **options):
        if name == "flint_decode_fast_activation":
            specification.update(options)
        return metal(name=name, **options)

    megakernel._metal = capture_specification
    engine, resident = build_decoder(model, optimized=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=True, fix_mistral_regex=False
    )
    text = "Write a complete Python program using openpyxl to create an Excel sales report, with sample data, formulas, formatting and a bar chart. Explain and test the code."
    prompt = tokenizer.apply_chat_template(
        [dict(role="user", content=text)],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
    )
    captured = {}
    persistent_kernel, greedy_kernel, filtered_run = (
        resident.kernel,
        resident.greedy_kernel,
        resident.filtered.run,
    )

    def persistent(**kwargs):
        out = persistent_kernel(**kwargs)
        captured["logits"], captured["trace"] = out[:2]
        return out

    def filtered(*a):
        out = filtered_run(*a)
        captured["logits"], captured["trace"] = out[3:5]
        return out

    def greedy(**kwargs):
        out = greedy_kernel(**kwargs)
        captured["logits"], captured["trace"] = out[:2]
        return out

    resident.kernel, resident.greedy_kernel, resident.filtered.run = persistent, greedy, filtered
    settings = [
        (0, 0.95, 20, 0),
        (0.05, 1, 0, 0),
        (0.6, 1, 0, 0),
        (1, 1, 0, 0),
        (1.7, 1, 0, 0),
        (2, 1, 0, 0),
        (1, 0.95, 0, 0),
        (0.6, 0.95, 20, 0),
        (1, 1, 1, 0),
        (2, 0.5, 149999, 0),
        (1, 0.01, 0, 0),
        (0, 1, 0, 1.5),
        (1, 1, 0, 1.5),
        (1, 0.95, 20, 1.5),
        (1, 0.95, 20, -1.5),
    ]
    results = []
    for context in (len(prompt), 513, 8192):
        ids = (prompt * ((context + len(prompt) - 1) // len(prompt)))[:context]
        for temperature, top_p, top_k, penalty in settings if context != 8192 else settings[:1]:
            for seed in (42, 73) if context != 8192 else (42,):
                steps = 520 if temperature == 0 and context == len(prompt) and seed == 42 else 16
                is_cuda = mx.device_info().get("architecture", "").startswith("sm_")
                sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
                unfiltered_t1 = (temperature, top_p, top_k) == (1, 1, 0)

                seen = set()

                def choose(logits, index):
                    flat = logits.reshape(-1).astype(mx.float32)
                    if penalty and seen:
                        flat[mx.array(list(seen), mx.int32)] -= penalty
                    if temperature == 0:
                        return int(mx.argmax(flat).item())
                    if unfiltered_t1:
                        noise = mx.random.gumbel(flat.shape, key=mx.random.key(seed + index))
                        return int(mx.argmax(flat + noise).item())
                    return int(sample(flat - mx.logsumexp(flat)).item())

                reference = engine.make_session(max_new_tokens=steps + 1)
                a = reference.prefill(ids)
                vocab_size = a.size
                a_np = np.array(a.view(mx.uint32))
                prefill_kv_digest = kv_digest(reference._state["kv"], reference._state["position"])
                mx.random.seed(seed)
                pending = choose(a, 0)
                first_pending = pending
                seen.add(pending)
                scalar_key, packet_key = mx.random.state[0], mx.random.state[0]
                del a
                # Shared global KV is mutated by decode; use a separate native reference on Metal.
                if not is_cuda:
                    native_session = engine.make_session(max_new_tokens=steps + 1)
                    native_session.prefill(ids)
                    native_cache = native_session.prompt_cache.layers
                    for row, cache in zip(engine.metadata[:49], native_cache, strict=True):
                        if not row[1]:
                            cache.grow(context + steps + 1)
                native_checks = 0
                position, packets = 0, 0
                recorded_packets = []
                while position < steps and pending not in model.eos_tokens:
                    count = min((1, 3, 8)[packets % 3], steps - position)
                    packet_input = pending
                    expected, logits, traces = [], [], []
                    mx.random.state[0] = scalar_key
                    for offset in range(count):
                        value, trace = engine.forward(reference._state, pending)
                        if not is_cuda and temperature == 0 and position + offset < 8:
                            h = model.native.model.embed_tokens(mx.array([[pending]], mx.int32))
                            native_trace = [h.reshape(-1)]
                            for layer, cache in zip(model.native.layers, native_cache, strict=True):
                                h = layer(h, None, cache)
                                native_trace.append(h.reshape(-1))
                            native_logits = model.native.lm_head(model.native.model.norm(h))
                            assert mx.array_equal(mx.stack(native_trace), trace).item(), (
                                "Native layer mismatch"
                            )
                            assert mx.array_equal(native_logits.reshape(-1), value).item(), (
                                "Native logits mismatch"
                            )
                            native_checks += 1
                        pending = choose(value, position + offset + 1)
                        seen.add(pending)
                        expected.append(pending)
                        logits.append(np.array(value.view(mx.uint32)))
                        traces.append(np.array(trace.view(mx.uint16)))
                        del value, trace
                        if pending in model.eos_tokens:
                            break
                    scalar_key = mx.random.state[0]
                    recorded_packets.append(
                        dict(
                            position=position,
                            count=count,
                            packet_input=packet_input,
                            expected=expected,
                            logits=logits,
                            traces=traces,
                            scalar_key=np.array(scalar_key),
                            seen=list(seen),
                            ref_pos=reference._state["position"],
                            ref_kv=kv_digest(reference._state["kv"], reference._state["position"]),
                        )
                    )
                    position += len(expected)
                    packets += 1
                if not is_cuda:
                    del native_session, native_cache
                del reference
                gc.collect()
                mx.synchronize()
                mx.clear_cache()

                candidate = engine.make_session(max_new_tokens=steps + 1)
                b = candidate.prefill(ids)
                assert np.array_equal(a_np, np.array(b.view(mx.uint32))), "Prefill logits differ"
                assert (
                    kv_digest(candidate._state["kv"], candidate._state["position"])
                    == prefill_kv_digest
                )
                del b
                if penalty:
                    candidate._state["presence_mask"] = mx.zeros(vocab_size, mx.float32)
                    candidate._state["presence_mask"][first_pending] = 1.0
                for pkt in recorded_packets:
                    mx.random.state[0] = packet_key
                    actual = resident.block(
                        candidate._state,
                        pkt["packet_input"],
                        pkt["count"],
                        seed + pkt["position"] + 1,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        presence_penalty=penalty,
                    )
                    packet_key = mx.random.state[0]
                    assert actual == pkt["expected"], (
                        context,
                        temperature,
                        top_p,
                        top_k,
                        pkt["position"],
                        actual,
                        pkt["expected"],
                    )
                    assert np.array_equal(np.array(packet_key), pkt["scalar_key"]), (
                        "Sampler key mismatch"
                    )
                    for offset in range(len(actual)):
                        assert np.array_equal(
                            np.array(captured["logits"][offset].view(mx.uint32)),
                            pkt["logits"][offset],
                        ), "Packet logits mismatch"
                        assert np.array_equal(
                            np.array(captured["trace"][offset].view(mx.uint16)),
                            pkt["traces"][offset],
                        ), "Packet layer mismatch"
                    if penalty:
                        mask = mx.zeros(vocab_size, mx.float32)
                        mask[mx.array(pkt["seen"], mx.int32)] = 1
                        assert mx.array_equal(candidate._state["presence_mask"], mask).item()
                    assert pkt["ref_pos"] == candidate._state["position"]
                    assert (
                        kv_digest(candidate._state["kv"], candidate._state["position"])
                        == pkt["ref_kv"]
                    ), "KV mismatch"
                row = dict(
                    context=context,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    presence_penalty=penalty,
                    seed=seed,
                    positions=position,
                    packets=packets,
                    native_checks=native_checks,
                )
                results.append(row)
                print(json.dumps(row), flush=True)
                captured.clear()
                del candidate, recorded_packets
                if top_k == 149999:
                    resident.filtered = FilteredResidentSampler(
                        resident.module, engine, specification
                    )
                    filtered_run = resident.filtered.run
                    resident.filtered.run = filtered
                gc.collect()
                mx.synchronize()
                mx.clear_cache()

    resident.kernel, resident.greedy_kernel, resident.filtered.run = (
        persistent_kernel,
        greedy_kernel,
        filtered_run,
    )
    stopping = validate_stopping(engine, resident, specification, tokenizer, prompt)
    print(json.dumps(dict(stopping=stopping)), flush=True)
    performance = []
    ids = prompt
    for temperature, top_p, top_k, penalty in (
        settings[0],
        settings[2],
        settings[3],
        settings[7],
        (1, 0.95, 20, 0),
        (1, 0.95, 20, 1.5),
    ):
        sample = make_sampler(temp=temperature, top_p=top_p, top_k=top_k)
        timings = {"scalar": [], "packet": []}
        for repeat in range(4):
            for variant in ("scalar", "packet") if repeat % 2 == 0 else ("packet", "scalar"):
                session = engine.make_session(max_new_tokens=257)
                logits = session.prefill(ids)
                mx.random.seed(42)
                if temperature == 0:
                    token = int(mx.argmax(logits).item())
                elif (temperature, top_p, top_k) == (1, 1, 0):
                    token = int(
                        mx.random.categorical(
                            logits.astype(mx.float32), key=mx.random.key(42)
                        ).item()
                    )
                else:
                    flat = logits.reshape(-1).astype(mx.float32)
                    token = int(sample(flat - mx.logsumexp(flat)).item())
                seen = {token}
                if penalty:
                    session._state["presence_mask"] = mx.zeros(logits.size, mx.float32)
                    session._state["presence_mask"][token] = 1.0
                count = 0
                mx.synchronize()
                started = time.perf_counter()
                while count < 256 and token not in model.eos_tokens:
                    if variant == "packet":
                        tokens = resident.block(
                            session._state,
                            token,
                            min(8, 256 - count),
                            43 + count,
                            temperature=temperature,
                            top_p=top_p,
                            top_k=top_k,
                            presence_penalty=penalty,
                        )
                        count += len(tokens)
                        token = tokens[-1]
                    else:
                        value = session.forward_token(token).reshape(-1).astype(mx.float32)
                        if penalty:
                            value[mx.array(list(seen), mx.int32)] -= penalty
                        if temperature == 0:
                            token = int(mx.argmax(value).item())
                        elif (temperature, top_p, top_k) == (1, 1, 0):
                            token = int(
                                mx.random.categorical(value, key=mx.random.key(43 + count)).item()
                            )
                        else:
                            token = int(sample(value - mx.logsumexp(value)).item())
                        seen.add(token)
                        count += 1
                mx.synchronize()
                speed = count / (time.perf_counter() - started)
                if repeat:
                    timings[variant].append(speed)
                del session, logits
                gc.collect()
                mx.synchronize()
                mx.clear_cache()
        row = dict(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            presence_penalty=penalty,
            **{name: statistics.median(values) for name, values in timings.items()},
        )
        performance.append(row)
        print(json.dumps(dict(performance=row)), flush=True)
    result = dict(
        device=mx.device_info(),
        config_sha256=hashlib.sha256((args.model / "config.json").read_bytes()).hexdigest(),
        cases=results,
        stopping=stopping,
        performance=performance,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
