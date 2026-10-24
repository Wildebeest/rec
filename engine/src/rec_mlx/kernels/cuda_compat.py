import os
import re
import sys
from pathlib import Path


def _ensure_cuda_path() -> None:
    if os.environ.get("CUDA_PATH") or os.environ.get("CUDA_HOME"):
        return
    candidates = [
        *(Path(p) / "nvidia" / "cu13" for p in sys.path if p),
        *(Path(p) / "nvidia" / "cuda_runtime" for p in sys.path if p),
        Path("/home/wilde/.unsloth/studio/unsloth_studio/lib/python3.13/site-packages/nvidia/cu13"),
        Path("/usr/local/cuda"),
    ]
    for candidate in candidates:
        if (candidate / "include" / "cuda_runtime.h").is_file():
            os.environ["CUDA_PATH"] = str(candidate)
            return


_ensure_cuda_path()

CUDA_COMPAT_HEADER = r"""
#include <cuda/atomic>

using uchar = unsigned char;
using ushort = unsigned short;
using uint = unsigned int;
using atomic_uint = unsigned int;
using bfloat16_t = __nv_bfloat16;

__device__ __forceinline__ float& operator+=(float& a, __nv_bfloat16 b) {
    a += __bfloat162float(b);
    return a;
}
__device__ __forceinline__ float operator/(__nv_bfloat16 a, float b) {
    return __bfloat162float(a) / b;
}
__device__ __forceinline__ __nv_bfloat16 operator+(int a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(float(a) + __bfloat162float(b));
}
__device__ __forceinline__ __nv_bfloat16 operator-(int a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(float(a) - __bfloat162float(b));
}
__device__ __forceinline__ __nv_bfloat16 operator/(int a, __nv_bfloat16 b) {
    return __float2bfloat16_rn(float(a) / __bfloat162float(b));
}
__device__ __forceinline__ bool operator<(__nv_bfloat16 a, int b) {
    return __bfloat162float(a) < float(b);
}
__device__ __forceinline__ bool operator>(__nv_bfloat16 a, int b) {
    return __bfloat162float(a) > float(b);
}

template <typename T, int N>
struct alignas(sizeof(T) * N) msl_vec {
    T data[N];
    __device__ __forceinline__ msl_vec() = default;
    __device__ __forceinline__ msl_vec(T a, T b, T c, T d) : data{a, b, c, d} {}
    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
};

struct alignas(8) msl_float2 {
    float x, y;
    __device__ __forceinline__ msl_float2() : x(0.0f), y(0.0f) {}
    __device__ __forceinline__ msl_float2(float s) : x(s), y(s) {}
    __device__ __forceinline__ msl_float2(float a, float b) : x(a), y(b) {}
};

struct alignas(16) msl_float4 {
    float data[4];
    __device__ __forceinline__ msl_float4() : data{0.0f, 0.0f, 0.0f, 0.0f} {}
    __device__ __forceinline__ msl_float4(float s) : data{s, s, s, s} {}
    __device__ __forceinline__ msl_float4(float a, float b, float c, float d) : data{a, b, c, d} {}
    __device__ __forceinline__ float& operator[](int i) { return data[i]; }
    __device__ __forceinline__ float operator[](int i) const { return data[i]; }
    __device__ __forceinline__ msl_float4& operator+=(const msl_float4& r) {
        data[0] += r.data[0];
        data[1] += r.data[1];
        data[2] += r.data[2];
        data[3] += r.data[3];
        return *this;
    }
};

__device__ __forceinline__ msl_float4 operator*(float s, const msl_float4& v) {
    return msl_float4(s * v.data[0], s * v.data[1], s * v.data[2], s * v.data[3]);
}

template <typename To, typename From>
__device__ __forceinline__ To as_type(From v);

template <>
__device__ __forceinline__ float as_type<float, uint>(uint v) {
    return __uint_as_float(v);
}

template <>
__device__ __forceinline__ uint as_type<uint, float>(float v) {
    return __float_as_uint(v);
}

namespace metal {
    __device__ __forceinline__ float abs(float x) { return fabsf(x); }
    __device__ __forceinline__ __nv_bfloat16 abs(__nv_bfloat16 x) { return __habs(x); }
    __device__ __forceinline__ float exp(float x) { return expf(x); }
    __device__ __forceinline__ __nv_bfloat16 exp(__nv_bfloat16 x) {
        return __float2bfloat16_rn(expf(__bfloat162float(x)));
    }
    __device__ __forceinline__ float exp2(float x) { return exp2f(x); }
    namespace precise {
        __device__ __forceinline__ float rsqrt(float x) { return rsqrtf(x); }
        __device__ __forceinline__ float exp(float x) { return expf(x); }
    }
    namespace fast {
        __device__ __forceinline__ float exp(float x) { return __expf(x); }
        __device__ __forceinline__ __nv_bfloat16 exp(__nv_bfloat16 x) {
            return __float2bfloat16_rn(expf(__bfloat162float(x)));
        }
        __device__ __forceinline__ float cos(float x) { return __cosf(x); }
        __device__ __forceinline__ float sin(float x) { return __sinf(x); }
    }
}

namespace mem_flags {
    enum Flag { mem_none = 0, mem_device = 1, mem_threadgroup = 2 };
}
enum msl_memory_order { memory_order_relaxed = 0, memory_order_seq_cst = 1 };
enum msl_thread_scope { thread_scope_device = 1 };

__device__ __forceinline__ void atomic_thread_fence(int, int, int) {
    cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_device);
}

__device__ __forceinline__ void threadgroup_barrier(int flags) {
    if (flags == mem_flags::mem_device) {
        cuda::atomic_thread_fence(cuda::memory_order_acq_rel, cuda::thread_scope_device);
    }
    __syncthreads();
}

__device__ __forceinline__ uint atomic_fetch_add_explicit(uint* ptr, uint val, int) {
    return cuda::atomic_ref<uint, cuda::thread_scope_device>(*ptr).fetch_add(
        val, cuda::memory_order_acq_rel
    );
}

__device__ __forceinline__ uint atomic_load_explicit(const uint* ptr, int) {
    return cuda::atomic_ref<uint, cuda::thread_scope_device>(*const_cast<uint*>(ptr)).load(
        cuda::memory_order_acquire
    );
}

__device__ __forceinline__ void atomic_store_explicit(uint* ptr, uint val, int) {
    cuda::atomic_ref<uint, cuda::thread_scope_device>(*ptr).store(
        val, cuda::memory_order_release
    );
}

__device__ __forceinline__ float simd_sum(float v) {
    #pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1)
        v += __shfl_xor_sync(0xffffffffu, v, mask);
    return v;
}

__device__ __forceinline__ msl_float2 simd_sum(msl_float2 v) {
    return msl_float2(simd_sum(v.x), simd_sum(v.y));
}

__device__ __forceinline__ float simd_max(float v) {
    #pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, mask));
    return v;
}

__device__ __forceinline__ uint simd_max(uint v) {
    #pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1) {
        uint o = __shfl_xor_sync(0xffffffffu, v, mask);
        v = o > v ? o : v;
    }
    return v;
}

__device__ __forceinline__ uint simd_min(uint v) {
    #pragma unroll
    for (int mask = 1; mask <= 16; mask <<= 1) {
        uint o = __shfl_xor_sync(0xffffffffu, v, mask);
        v = o < v ? o : v;
    }
    return v;
}

__device__ __forceinline__ float simd_shuffle(float v, int lane) {
    return __shfl_sync(0xffffffffu, v, lane);
}
__device__ __forceinline__ uint simd_shuffle(uint v, int lane) {
    return __shfl_sync(0xffffffffu, v, lane);
}
__device__ __forceinline__ float simd_shuffle_down(float v, int delta) {
    return __shfl_down_sync(0xffffffffu, v, delta);
}
__device__ __forceinline__ uint simd_broadcast(uint v, int lane) {
    return __shfl_sync(0xffffffffu, v, lane);
}

#define vec msl_vec
#define float2 msl_float2
#define float4 msl_float4
#define device
#define threadgroup
#define thread
#define thread_index_in_threadgroup uint(threadIdx.x)
#define thread_index_in_simdgroup uint(threadIdx.x & 31u)
#define simdgroup_index_in_threadgroup uint(threadIdx.x >> 5)
#define threadgroup_position_in_grid blockIdx
#define thread_position_in_grid make_uint3(blockIdx.x * blockDim.x + threadIdx.x, 0u, 0u)
"""


def _translate_pragmas(text: str) -> str:
    text = re.sub(r"#pragma\s+clang\s+loop\s+unroll_count\((\d+)\)", r"#pragma unroll \1", text)
    text = re.sub(r"#pragma\s+clang\s+loop\s+unroll\(full\)", r"#pragma unroll", text)
    text = re.sub(r"#pragma\s+clang\s+loop\s+unroll\(disable\)", r"#pragma unroll 1", text)
    return text


def translate_msl_to_cuda(header: str, source: str) -> tuple[str, str]:
    header = _translate_pragmas(header)
    source = _translate_pragmas(source)

    # Remove the threadgroup-address-space overload of weight_matrix which is identical in CUDA.
    header = re.sub(
        r"inline\s+Matrix\s+weight_matrix\(const\s+device\s+uchar\*\s*bank,\s*const\s+threadgroup\s+uint\*\s*d\)\s*\{\s*return\s*\{[^}]+\};\s*\}",
        "",
        header,
    )

    # Annotate KVView methods and inline helper functions for CUDA device execution.
    for method in (
        "KVView head(",
        "KVView value(",
        "const device bfloat16_t* address(",
        "bfloat16_t operator[](",
        "void store(",
    ):
        header = header.replace(method, "__device__ __forceinline__ " + method)
    header = re.sub(r"\binline\b", "__device__ __forceinline__", header)

    # Convert local threadgroup array declarations in kernel body to __shared__.
    source = re.sub(
        r"\b(alignas\(\d+\)\s+)?threadgroup\s+(uint|float|bfloat16_t)\s+(\w+\s*\[[^\]]+\]\s*;)",
        r"__shared__ \1\2 \3",
        source,
    )

    # Vectorize 8-byte weight loads in dot16 and dot16_w197_bias23 via read-only cache (__ldg).
    header = header.replace(
        "const device ushort* p=(const device ushort*)w;",
        "const uint2 _w2=__ldg((const uint2*)w);const uint p[4]={_w2.x&0xffffu,_w2.x>>16,_w2.y&0xffffu,_w2.y>>16};",
    )

    # Vectorize 8-byte activation loads in load16.
    header = header.replace(
        "inline float load16(const device bfloat16_t* x, thread float* v) {\n"
        "    float sum=0.0f;\n"
        "    for(uint i=0;i<16;i+=4) {\n"
        "        sum+=x[i]+x[i+1]+x[i+2]+x[i+3];\n"
        "        v[i]=x[i];v[i+1]=x[i+1]/16.0f;\n"
        "        v[i+2]=x[i+2]/256.0f;v[i+3]=x[i+3]/4096.0f;\n"
        "    }\n"
        "    return sum;\n"
        "}",
        "inline float load16(const device bfloat16_t* x, thread float* v) {\n"
        "    float sum=0.0f;\n"
        "    #pragma unroll\n"
        "    for(uint i=0;i<16;i+=4) {\n"
        "        const uint2 _x2=__ldg((const uint2*)(x+i));\n"
        "        const bfloat16_t* _xp=(const bfloat16_t*)&_x2;\n"
        "        sum+=_xp[0]+_xp[1]+_xp[2]+_xp[3];\n"
        "        v[i]=_xp[0];v[i+1]=_xp[1]/16.0f;\n"
        "        v[i+2]=_xp[2]/256.0f;v[i+3]=_xp[3]/4096.0f;\n"
        "    }\n"
        "    return sum;\n"
        "}",
    )

    # Eliminate redundant back-to-back device fences in barrier helpers.
    header = header.replace(
        "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);\n"
        "    threadgroup_barrier(mem_flags::mem_device);",
        "threadgroup_barrier(mem_flags::mem_device);",
    )

    # Widen expert_team_gateup from 2 rows/iter to 4 rows/iter to halve load16_local calls.
    header = header.replace(
        "    for(uint local=role_sg*2;local<width;local+=teams*8*2) {\n"
        "        float ga[2]={0.0f,0.0f},ua[2]={0.0f,0.0f};\n"
        "        for(uint b=0;b<2048;b+=512) {\n"
        "            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);\n"
        "            for(uint j=0;j<2;++j) {\n"
        "                uint row=expert*g.n+local+j,gi=row*32+scalar/64;\n"
        "                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(g.s[gi]),float(g.b[gi]),sum);\n"
        "                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(u.s[gi]),float(u.b[gi]),sum);\n"
        "            }\n"
        "        }\n"
        "        for(uint j=0;j<2;++j) {",
        "    for(uint local=role_sg*4;local<width;local+=teams*8*4) {\n"
        "        float ga[4]={0.0f,0.0f,0.0f,0.0f},ua[4]={0.0f,0.0f,0.0f,0.0f};\n"
        "        for(uint b=0;b<2048;b+=512) {\n"
        "            uint scalar=b+lane*16;float v[16];float sum=load16_local(x+scalar,v);\n"
        "            #pragma unroll\n"
        "            for(uint j=0;j<4;++j) {\n"
        "                uint row=expert*width+local+j,gi=row*32+scalar/64;\n"
        "                ga[j]+=dot16(g.w+size_t(row)*1024+scalar/2,v,float(__ldg(&g.s[gi])),float(__ldg(&g.b[gi])),sum);\n"
        "                ua[j]+=dot16(u.w+size_t(row)*1024+scalar/2,v,float(__ldg(&u.s[gi])),float(__ldg(&u.b[gi])),sum);\n"
        "            }\n"
        "        }\n"
        "        #pragma unroll\n"
        "        for(uint j=0;j<4;++j) {",
    )

    # Hoist loop-invariant load16 out of the row loop in expert_team_down and specialize k=512/1024.
    header = header.replace(
        "    for(uint local=role_sg*4;local<2048;local+=teams*8*4) {\n"
        "        float a[4]={0.0f,0.0f,0.0f,0.0f};\n"
        "        for(uint b=0;b<m.k;b+=512) {\n"
        "            uint scalar=b+lane*16;float v[16];float sum=load16(input+scalar,v);\n"
        "            for(uint j=0;j<4;++j) {\n"
        "                uint row=expert*2048+local+j,gi=row*(m.k/64)+scalar/64;\n"
        "                a[j]+=dot16_w197_bias23(m.w+size_t(row)*(m.k/2)+scalar/2,v,float(m.s[gi]),float(m.b[gi]),sum);\n"
        "            }\n"
        "        }\n"
        "        for(uint j=0;j<4;++j){float v=simd_sum(a[j]);if(lane==0)out[role*2048+local+j]=bfloat16_t(v);}\n"
        "    }",
        "    if(!shared) {\n"
        "        uint scalar=lane*16;float v[16];float sum=load16(input+scalar,v);\n"
        "        for(uint local=role_sg*4;local<2048;local+=teams*8*4) {\n"
        "            float a[4]={0.0f,0.0f,0.0f,0.0f};\n"
        "            #pragma unroll\n"
        "            for(uint j=0;j<4;++j) {\n"
        "                uint row=expert*2048+local+j,gi=row*8+scalar/64;\n"
        "                a[j]+=dot16_w197_bias23(m.w+size_t(row)*256+scalar/2,v,float(__ldg(&m.s[gi])),float(__ldg(&m.b[gi])),sum);\n"
        "            }\n"
        "            #pragma unroll\n"
        "            for(uint j=0;j<4;++j){float val=simd_sum(a[j]);if(lane==0)out[role*2048+local+j]=bfloat16_t(val);}\n"
        "        }\n"
        "    } else {\n"
        "        float v0[16],v1[16];\n"
        "        float sum0=load16(input+lane*16,v0),sum1=load16(input+512+lane*16,v1);\n"
        "        for(uint local=role_sg*4;local<2048;local+=teams*8*4) {\n"
        "            float a[4]={0.0f,0.0f,0.0f,0.0f};\n"
        "            #pragma unroll\n"
        "            for(uint j=0;j<4;++j) {\n"
        "                uint row=local+j,gi0=row*16+lane/4,gi1=gi0+8;\n"
        "                a[j]+=dot16_w197_bias23(m.w+size_t(row)*512+lane*8,v0,float(__ldg(&m.s[gi0])),float(__ldg(&m.b[gi0])),sum0);\n"
        "                a[j]+=dot16_w197_bias23(m.w+size_t(row)*512+256+lane*8,v1,float(__ldg(&m.s[gi1])),float(__ldg(&m.b[gi1])),sum1);\n"
        "            }\n"
        "            #pragma unroll\n"
        "            for(uint j=0;j<4;++j){float val=simd_sum(a[j]);if(lane==0)out[role*2048+local+j]=bfloat16_t(val);}\n"
        "        }\n"
        "    }",
    )

    # Distribute q4 (o_proj, m.n=2048) at 2 rows/warp so 1024 warps (128 SMs) participate instead of 512 warps (64 SMs).
    header = header.replace(
        "template<uint K>\n"
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
        "template<uint K>\n"
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
    )

    # Use read-only cache (__ldg) for quantization scales and biases in q4_local and q4_local_block.
    header = header.replace(
        "float(m.s[gi]),float(m.b[gi])",
        "float(__ldg(&m.s[gi])),float(__ldg(&m.b[gi]))",
    )

    # Coalesce 4-element bfloat16 KV loads in attention_partial_path_w53 into 64-bit __ldg(uint2*).
    header = header.replace(
        "            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(key_base[at+j]);value[j]=float(value_base[at+j]);}",
        "            bfloat16_t rk[4],rv[4];\n"
        "            *((uint2*)rk)=__ldg((const uint2*)key_base.address(at));\n"
        "            *((uint2*)rv)=__ldg((const uint2*)value_base.address(at));\n"
        "            float key[4],value[4];for(uint j=0;j<4;++j){key[j]=float(rk[j]);value[j]=float(rv[j]);}",
    )
    header = header.replace(
        "        if(token<count)for(uint j=0;j<4;++j) {\n"
        "            uint at=uint(token)*128+j;\n"
        "            next_key[j]=key_base[at];next_value[j]=value_base[at];\n"
        "        }",
        "        if(token<count) {\n"
        "            uint at=uint(token)*128;\n"
        "            *((uint2*)next_key)=__ldg((const uint2*)key_base.address(at));\n"
        "            *((uint2*)next_value)=__ldg((const uint2*)value_base.address(at));\n"
        "        }",
    )
    header = header.replace(
        "            if(token+parts<count)for(uint j=0;j<4;++j) {\n"
        "                uint next=at+parts*128+j;\n"
        "                next_key[j]=key_base[next];next_value[j]=value_base[next];\n"
        "            }",
        "            if(token+parts<count) {\n"
        "                uint next=at+parts*128;\n"
        "                *((uint2*)next_key)=__ldg((const uint2*)key_base.address(next));\n"
        "                *((uint2*)next_value)=__ldg((const uint2*)value_base.address(next));\n"
        "            }",
    )

    # Scalar bool input 'active' in FilteredResidentSampler is passed as const bool* in CUDA.
    source = source.replace("if(!active)", "if(!(*active))")

    return CUDA_COMPAT_HEADER + "\n" + header + "\n", source
