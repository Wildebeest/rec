import json

from .kv_teams import function

ORIGINAL = "else {auto tail=1/(1+metal::exp(metal::abs(g)));bfloat16_t sig=g<0?tail:1-tail;out[role*512+local+j]=bfloat16_t(bfloat16_t(g*sig)*u);}"
CHANGED = ORIGINAL.replace("metal::exp(", "metal::fast::exp(")


def transform(header):
    old = function(header, "expert_team_gateup")
    assert old.count(ORIGINAL) == 1
    new = old.replace(ORIGINAL, CHANGED)
    result = header.replace(old, new)
    assert result.replace(CHANGED, ORIGINAL) == header
    return result


def guard(module, engine, spec):
    mx = module.mx
    rows = [d for d in engine.metadata[:49] if d[2]]
    assert len(rows) == 47
    assert all(
        tuple(d[i] for i in (33, 34, 38, 39, 48, 49, 53, 54))
        == (512, 2048, 512, 2048, 1024, 2048, 1024, 2048)
        for d in rows
    )
    assert function(spec["header"], "expert_team_gateup").count(ORIGINAL) == 1
    kernel = module._metal(
        name="routed_fast_exp_boundary_guard",
        input_names=["gates", "ups"],
        output_names=["reference", "candidate", "tails"],
        source=r"""
        uint i=thread_position_in_grid.x;
        if(i<1048576u) {
            uint gate=i&65535u;
            bfloat16_t g=gates[gate],u=ups[i>>16];
            auto tail=1/(1+metal::exp(metal::abs(g)));
            auto fast_tail=1/(1+metal::fast::exp(metal::abs(g)));
            bfloat16_t sig=g<0?tail:1-tail;
            bfloat16_t fast_sig=g<0?fast_tail:1-fast_tail;
            bfloat16_t activated=bfloat16_t(g*sig);
            bfloat16_t fast_activated=bfloat16_t(g*fast_sig);
            reference[i]=as_type<uint>(float(bfloat16_t(activated*u)))>>16;
            candidate[i]=as_type<uint>(float(bfloat16_t(fast_activated*u)))>>16;
            if(i<65536u) {
                reference[1048576u+i]=(as_type<uint>(float(sig))>>16)|
                    (as_type<uint>(float(activated))&0xffff0000u);
                candidate[1048576u+i]=(as_type<uint>(float(fast_sig))>>16)|
                    (as_type<uint>(float(fast_activated))&0xffff0000u);
                tails[i]=as_type<uint>(float(tail));
                tails[65536u+i]=as_type<uint>(float(fast_tail));
            }
        }
        """,
        compile_options={"math_mode": "safe"},
    )
    bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
    up_bits = [
        0,
        0x8000,
        1,
        0x8001,
        0x007F,
        0x807F,
        0x0080,
        0x8080,
        0x3F80,
        0xBF80,
        0x7F7F,
        0xFF7F,
        0x7F80,
        0xFF80,
        0x7FC1,
        0xFF81,
    ]
    out = kernel(
        inputs=[
            bits.view(mx.bfloat16),
            mx.array(up_bits, dtype=mx.uint16).view(mx.bfloat16),
        ],
        grid=(1048576, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1114112,), (1114112,), (131072,)],
        output_dtypes=[mx.uint32, mx.uint32, mx.uint32],
    )
    mx.eval(*out)
    reference, candidate, tails = (v.tolist() for v in out)
    product_errors = sum(
        a != b for a, b in zip(reference[:1048576], candidate[:1048576], strict=False)
    )
    boundary = [
        (i, a, b)
        for i, (a, b) in enumerate(zip(reference[1048576:], candidate[1048576:], strict=False))
        if a != b
    ]
    finite_errors = sum((i & 0x7F80) != 0x7F80 for i, _, _ in boundary)
    print(
        "ROUTED_FAST_EXP_GUARD "
        + json.dumps(
            {
                "scored": False,
                "gate_patterns": 65536,
                "up_patterns": up_bits,
                "product_cases": 1048576,
                "product_errors": product_errors,
                "boundary_errors": len(boundary),
                "finite_gate_boundary_errors": finite_errors,
                "first_boundary_differences": boundary[:16],
                "raw_tail_bit_differences": sum(
                    a != b for a, b in zip(tails[:65536], tails[65536:], strict=False)
                ),
                "device": mx.device_info(),
                "qualified_architecture": mx.device_info()["architecture"],
            }
        ),
        flush=True,
    )
    assert product_errors == len(boundary) == 0, (
        product_errors,
        len(boundary),
        finite_errors,
    )
