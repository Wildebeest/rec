from .kv_teams import function, once

Q3_DOT = r"""
inline float qat_dot3(const device uchar* w,thread float* x,float s,float b,float sum) {
    const device ushort* words=(const device ushort*)w;
    uint lo=words[0],mid=words[1],hi=words[2];
    uint codes[4]={lo&4095u,((lo>>12)|(mid<<4))&4095u,
                   ((mid>>8)|(hi<<8))&4095u,hi>>4};
    float a=0.0f;
    #pragma clang loop unroll(full)
    for(uint j=0;j<4;++j) {
        uint p=codes[j];
        // Integer payloads below 2^23 have an exact unit-spaced FP32 encoding.
        a+=x[4*j]*(as_type<float>(0x4b000000u|(p&7u))-8388608.0f)
          +x[4*j+1]*(as_type<float>(0x4b000000u|((p&56u)<<1))-8388608.0f)
          +x[4*j+2]*(as_type<float>(0x4b000000u|((p&448u)<<2))-8388608.0f)
          +x[4*j+3]*(as_type<float>(0x4b000000u|((p&3584u)<<3))-8388608.0f);
    }
    return s*a+sum*b;
}
"""


def transform(header, body, model, mlp_workers, specialize_mixed_experts):
    assert model.embedding.bits == 8 and model.head.bits == 4
    bits = model.config["mlx_export"]["routed_expert_bits"]
    body = once(
        body,
        "size_t row=uint(token);uint code=(embedding.w[row*1024+i/2]>>((i%2)*4))&15;",
        "size_t row=uint(token);uint code=embedding.w[row*2048+i];",
    )
    old = function(header, "route_dot_bf16")
    new = once(
        old,
        "if(lane==0){float tail=",
        "a=float(bfloat16_t(a));\n        if(lane==0){float tail=",
    )
    header = header.replace(old, new)
    # QAT casts routing weights and each weighted expert product to BF16.
    score0 = "simd_shuffle(w3_weight,ushort(0))" if "w3_weight" in body else "scores[0]"
    scorer = "simd_shuffle(w3_weight,ushort(r))" if "w3_weight" in body else "scores[r]"
    body = once(
        body,
        f"float(down[i])*{score0}+0.0f",
        f"float(bfloat16_t(down[i]*bfloat16_t({score0})))+0.0f",
    )
    body = once(
        body,
        f"float(down[r*2048+i])*{scorer}+0.0f",
        f"float(bfloat16_t(down[r*2048+i]*bfloat16_t({scorer})))+0.0f",
    )
    # MLX's strided six-expert sum rounds to BF16 after every addition.
    mixed = f"(float(bfloat16_t(down[r*2048+i]*bfloat16_t({scorer})))+0.0f)+sums[j]"
    body = once(body, f"sums[j]={mixed};", f"sums[j]=float(bfloat16_t({mixed}));")
    if bits == 3:
        shared_at = mlp_workers * 3 // 4
        header = once(
            header,
            "inline void expert_team_gateup",
            Q3_DOT + "\ninline void expert_team_gateup",
        )
        old = function(header, "expert_team_gateup")
        new = once(
            old,
            "ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);",
            "ga[j]+=shared?dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum):"
            "qat_dot3(g.w+size_t(row)*768+scalar*3/8,v,float(g.s[gi]),float(g.b[gi]),sum);",
        )
        new = once(
            new,
            "ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);",
            "ua[j]+=shared?dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum):"
            "qat_dot3(u.w+size_t(row)*768+scalar*3/8,v,float(u.s[gi]),float(u.b[gi]),sum);",
        )
        header = header.replace(old, new)
        if specialize_mixed_experts:
            old = function(header, "expert_team_gateup")
            new = once(old, f"bool shared=group>={shared_at};", "constexpr bool shared=SHARED;")
            header = header.replace(old, "template<bool SHARED>\n" + new)
            header += f"""
inline void qat_dispatch_gateup(const device uchar* bank,const threadgroup uint* d,
    const threadgroup bfloat16_t* x,const device uint* ids,const device bfloat16_t* silu,
    device bfloat16_t* out,uint lane,uint local_sg,uint group) {{
    if(group>={shared_at})expert_team_gateup<true>(bank,d,x,ids,silu,out,lane,local_sg,group);
    else expert_team_gateup<false>(bank,d,x,ids,silu,out,lane,local_sg,group);
}}
"""
        old = function(header, "expert_team_down")
        new = once(
            old,
            "a[j]+=dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);",
            "a[j]+=shared?dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum):"
            "qat_dot3(m.w+size_t(row)*(m.k*3/8)+scalar*3/8,v,float(m.s[gi]),float(m.b[gi]),sum);",
        )
        header = header.replace(old, new)
        if specialize_mixed_experts:
            old = function(header, "expert_team_down")
            new = once(old, f"bool shared=group>={shared_at};", "constexpr bool shared=SHARED;")
            header = header.replace(old, "template<bool SHARED>\n" + new)
            header += f"""
inline void qat_dispatch_down(const device uchar* bank,const threadgroup uint* d,
    const device bfloat16_t* x,const device uint* ids,device bfloat16_t* out,
    uint lane,uint local_sg,uint group) {{
    if(group>={shared_at})expert_team_down<true>(bank,d,x,ids,out,lane,local_sg,group);
    else expert_team_down<false>(bank,d,x,ids,out,lane,local_sg,group);
}}
"""
            body = once(body, "expert_team_gateup(bank,d,", "qat_dispatch_gateup(bank,d,")
            body = once(body, "expert_team_down(bank,d,", "qat_dispatch_down(bank,d,")
    return header, body


def install(module, model, mlp_workers, specialize_mixed_experts):
    original = module._metal

    def build(name, **options):
        if name == "flint_decode":
            header, source = transform(
                options["header"], options["source"], model, mlp_workers, specialize_mixed_experts
            )
            options = dict(options, header=header, source=source)
        return original(name, **options)

    module._metal = build
