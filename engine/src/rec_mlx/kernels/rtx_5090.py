from .kv_teams import function, once


def transform(header: str, source: str) -> tuple[str, str]:
    # Vectorize 8-byte weight loads in dot16 and dot16_w197_bias23 via read-only cache (__ldg).
    for name in ("dot16", "dot16_w197_bias23"):
        original = function(header, name)
        header = header.replace(
            original,
            once(
                original,
                "const device ushort* p=(const device ushort*)w;",
                "const uint2 _w2=__ldg((const uint2*)w);const uint p[4]={_w2.x&0xffffu,_w2.x>>16,_w2.y&0xffffu,_w2.y>>16};",
            ),
        )

    # Vectorize 8-byte activation loads in load16.
    original = function(header, "load16")
    header = header.replace(
        original,
        once(
            original,
            "    float sum=0.0f;\n"
            "    for(uint i=0;i<16;i+=4) {\n"
            "        sum+=x[i]+x[i+1]+x[i+2]+x[i+3];\n"
            "        v[i]=x[i];v[i+1]=x[i+1]/16.0f;\n"
            "        v[i+2]=x[i+2]/256.0f;v[i+3]=x[i+3]/4096.0f;\n"
            "    }\n",
            "    float sum=0.0f;\n"
            "    #pragma unroll\n"
            "    for(uint i=0;i<16;i+=4) {\n"
            "        const uint2 _x2=__ldg((const uint2*)(x+i));\n"
            "        const bfloat16_t* _xp=(const bfloat16_t*)&_x2;\n"
            "        sum+=_xp[0]+_xp[1]+_xp[2]+_xp[3];\n"
            "        v[i]=_xp[0];v[i+1]=_xp[1]/16.0f;\n"
            "        v[i+2]=_xp[2]/256.0f;v[i+3]=_xp[3]/4096.0f;\n"
            "    }\n",
        ),
    )

    # Distribute q4 (o_proj, m.n=2048) at 2 rows/warp so 1024 warps (128 SMs) participate instead of 512 warps (64 SMs).
    original = function(header, "q4")
    header = header.replace(
        original,
        once(
            original,
            "inline void q4(Matrix m,const device bfloat16_t* x,device bfloat16_t* out,\n"
            "               uint lane,uint sg,uint groups,uint expert=0) {\n"
            "    for(uint first=sg*4;first<m.n;first+=groups*4) {\n"
            "        float a[4]={0.0f,0.0f,0.0f,0.0f};\n"
            "        for(uint block=0;block<K;block+=512) {\n"
            "            uint scalar=block+lane*16;\n"
            "            float v[16];float sum=load16(x+scalar,v);\n"
            "            for(uint r=0;r<4;++r) {\n"
            "                uint row=expert*m.n+first+r;uint gi=row*(K/64)+scalar/64;\n"
            "                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);\n"
            "            }\n"
            "        }\n"
            "        for(uint r=0;r<4;++r) {\n"
            "            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);\n"
            "        }\n"
            "    }\n"
            "}",
            "inline void q4(Matrix m,const device bfloat16_t* x,device bfloat16_t* out,\n"
            "               uint lane,uint sg,uint groups,uint expert=0) {\n"
            "    for(uint first=sg*2;first<m.n;first+=groups*2) {\n"
            "        float a[2]={0.0f,0.0f};\n"
            "        for(uint block=0;block<K;block+=512) {\n"
            "            uint scalar=block+lane*16;\n"
            "            float v[16];float sum=load16(x+scalar,v);\n"
            "            #pragma unroll\n"
            "            for(uint r=0;r<2;++r) {\n"
            "                uint row=expert*m.n+first+r;uint gi=row*(K/64)+scalar/64;\n"
            "                a[r]+=dot16(m.w+size_t(row)*(K/2)+scalar/2,v,float(__ldg(&m.s[gi])),float(__ldg(&m.b[gi])),sum);\n"
            "            }\n"
            "        }\n"
            "        #pragma unroll\n"
            "        for(uint r=0;r<2;++r) {\n"
            "            float v=simd_sum(a[r]);if(lane==0)out[first+r]=bfloat16_t(v);\n"
            "        }\n"
            "    }\n"
            "}",
        ),
    )

    # Use read-only cache (__ldg) for quantization scales and biases in q4_local, q4_local_block and expert_team_down.
    for name in ("q4_local", "q4_local_block", "expert_team_down"):
        if f"{name}(" not in header:
            continue
        original = function(header, name)
        count = 2 if name == "expert_team_down" and "qat_dot3" in original else 1
        header = header.replace(
            original,
            once(
                original,
                "float(m.s[gi]),float(m.b[gi])",
                "float(__ldg(&m.s[gi])),float(__ldg(&m.b[gi]))",
                count=count,
            ),
        )

    # Coalesce 4-element bfloat16 KV loads in attention_partial_path_w53 into 64-bit __ldg(uint2*),
    # specialize direct p0 pointer arithmetic for !ROUNDED (all 34 SWA layers),
    # share K/V address calculation, and avoid dynamic register-array indexing.
    original = function(header, "attention_partial_path_w53")
    replacement = original
    replacement = once(
        replacement,
        "            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}",
        "            bfloat16_t rk[4],rv[4];\n"
        "            const device bfloat16_t* kp=key_base.address(at);\n"
        "            *((uint2*)rk)=__ldg((const uint2*)kp);\n"
        "            *((uint2*)rv)=__ldg((const uint2*)(kp+cap*512u));\n"
        "            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(rk[j]);value[j]=float(rv[j]);}",
    )
    replacement = once(
        replacement,
        "        if(token<count)for(uint j=0;j<4;++j) {\n"
        "            uint at=uint(token)*128+j;\n"
        "            next_key[j]=key_base[at];next_value[j]=value_base[at];\n"
        "        }",
        "        if(token<count) {\n"
        "            uint at=uint(token)*128;\n"
        "            const device bfloat16_t* kp=ROUNDED?key_base.address(at):(key_base.p0+key_base.offset+at);\n"
        "            *((uint2*)next_key)=__ldg((const uint2*)kp);\n"
        "            *((uint2*)next_value)=__ldg((const uint2*)(kp+cap*512u));\n"
        "        }",
    )
    replacement = once(
        replacement,
        "            float key[4];bfloat16_t value[4];\n"
        "            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=next_value[j];}\n"
        "            if(token+parts<count)for(uint j=0;j<4;++j) {\n"
        "                uint next=at+parts*128+j;\n"
        "                next_key[j]=key_base[next];next_value[j]=value_base[next];\n"
        "            }\n"
        "            float scores[HEADS];\n"
        "            for(uint h=0;h<HEADS;++h) {\n"
        "                float score=0.0f;for(uint j=0;j<4;++j)score+=query[h][j]*key[j];\n"
        "                scores[h]=simd_sum(score);\n"
        "            }\n"
        "            float old=0.0f,add=0.0f;\n"
        "            if(lane<HEADS) {\n"
        "                float score=scores[lane],next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));\n"
        "                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;\n"
        "                z=z*old+add;hi=next;\n"
        "            }",
        "            float key[4],value[4];\n"
        "            #pragma unroll\n"
        "            for(uint j=0;j<4;++j){key[j]=float(next_key[j]);value[j]=float(next_value[j]);}\n"
        "            if(token+parts<count) {\n"
        "                uint next=at+parts*128;\n"
        "                const device bfloat16_t* kp=ROUNDED?key_base.address(next):(key_base.p0+key_base.offset+next);\n"
        "                *((uint2*)next_key)=__ldg((const uint2*)kp);\n"
        "                *((uint2*)next_value)=__ldg((const uint2*)(kp+cap*512u));\n"
        "            }\n"
        "            float my_score=0.0f;\n"
        "            #pragma unroll\n"
        "            for(uint h=0;h<HEADS;++h) {\n"
        "                float score=0.0f;\n"
        "                #pragma unroll\n"
        "                for(uint j=0;j<4;++j)score+=query[h][j]*key[j];\n"
        "                score=simd_sum(score);\n"
        "                if(lane==h)my_score=score;\n"
        "            }\n"
        "            float old=0.0f,add=0.0f;\n"
        "            if(lane<HEADS) {\n"
        "                float score=my_score,next=max(hi,score),low=metal::fast::exp(-metal::abs(score-hi));\n"
        "                bool wins=score>hi;old=wins?low:1.0f;add=wins?1.0f:low;\n"
        "                z=z*old+add;hi=next;\n"
        "            }",
    )
    replacement = once(
        replacement,
        "            for(uint h=0;h<HEADS;++h) {\n"
        "                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));\n"
        "                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*float(value[j]);\n"
        "            }",
        "            for(uint h=0;h<HEADS;++h) {\n"
        "                float factor=simd_shuffle(old,ushort(h)),weight=simd_shuffle(add,ushort(h));\n"
        "                #pragma unroll\n"
        "                for(uint j=0;j<4;++j)acc[h][j]=acc[h][j]*factor+weight*value[j];\n"
        "            }",
    )
    header = header.replace(original, replacement)

    # Align long-context global attention partition counts (parts) to 160-SM (1280-warp / 320-warp-per-KV-team) waves on RTX 5090.
    source = once(
        source,
        "uint parts=rounded?(count==1024?64:count<=8192?128:(count<=32768?256:(count<=65536?512:1024))):32;",
        "uint parts=rounded?(count==1024?64:count<=16384?320:(count<=65536?640:960)):32;",
    )
    return header, source
