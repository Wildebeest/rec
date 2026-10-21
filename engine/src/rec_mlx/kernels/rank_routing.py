ROUTE_CALL = "route_top_lane_scores<true>((const device float*)(bank+d[9]),statistics,ids,scores,lane,simdgroup_index_in_threadgroup);"
OWN_CALL = "w3_route_own_rank((const device float*)(bank+d[9]),statistics,(device float*)(routing+8),ids,lane,simdgroup_index_in_threadgroup,group);"
# Group zero's six score slots survive the next layer's team-local attention.
# The next router's global join completes all residual reads before reuse.
MIX = """            float w3_chosen=lane<6?((device float*)(routing+8))[lane]:0.0f;
            float w3_sum=0.0f;
            for(uint j=0;j<6;++j)w3_sum=simd_shuffle(w3_chosen,j)+w3_sum;
            w3_sum+=1.0e-20f;
            float w3_weight=lane<6?w3_chosen/w3_sum:0.0f;
"""
HELPER = r"""
inline void w3_route_own_rank(const device float* bias,device float* probs,
                            device float* chosen,device uint* ids,uint lane,uint sg,uint group) {
    if(sg==0 && group<24) {
        uint role=group/4;
        float p[8],b[8];for(uint j=0;j<8;++j){p[j]=probs[lane+32*j];b[j]=p[j]+bias[lane+32*j];}
        for(uint rank=0;rank<=role;++rank) {
            float best=-3.402823466e+38f;uint idx=0;
            for(uint j=0;j<8;++j){uint i=lane+32*j;if(b[j]>best||(b[j]==best&&i<idx)){best=b[j];idx=i;}}
            float peak=simd_max(best);uint selected=simd_min(best==peak?idx:0xffffffffu);
            if(rank==role) {
                float value=simd_shuffle(p[selected/32],selected%32);
                if(lane==0) {
                    ids[role]=selected;
                    if(group%4==0)chosen[role]=value;
                }
            }
            for(uint j=0;j<8;++j)if(lane+32*j==selected)b[j]=-3.402823466e+38f;
        }
    }
    threadgroup_barrier(mem_flags::mem_device);
}
"""


def transform(header, body, mlp_workers=32):
    assert body.count(ROUTE_CALL) == 1
    assert body.count("            float sums[8];") == 1
    assert body.count("*scores[0]") == body.count("*scores[r]") == 1
    body = body.replace(ROUTE_CALL, OWN_CALL)
    body = body.replace("            float sums[8];", MIX + "            float sums[8];")
    body = body.replace("*scores[0]", "*simd_shuffle(w3_weight,ushort(0))")
    body = body.replace("*scores[r]", "*simd_shuffle(w3_weight,ushort(r))")
    team = mlp_workers // 8
    helper = HELPER.replace("group<24", f"group<{team * 6}")
    helper = helper.replace("group/4", f"group/{team}").replace("group%4", f"group%{team}")
    return header + helper, body


def install(module):
    factory = module._metal
    captures = {}

    def metal(name, *, source, header="", **options):
        if name == "flint_decode":
            captures["parent"] = {
                "name": name,
                "source": source,
                "header": header,
                "options": options,
            }
            header, source = transform(header, source)
            captures["rank"] = {
                "name": name,
                "source": source,
                "header": header,
                "options": options,
            }
        return factory(name=name, source=source, header=header, **options)

    module._metal = metal
    return factory, captures
