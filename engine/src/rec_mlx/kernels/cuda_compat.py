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
    template <typename T> __device__ __forceinline__ T clamp(T x, T lo, T hi) { return x < lo ? lo : (x > hi ? hi : x); }
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

    header = re.sub(
        r"\bconstant\s+((?:float|uint|int|ushort|uchar|bfloat16_t)\s+\w+\s*\[)",
        r"__device__ const \1",
        header,
    )

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

    # Eliminate redundant back-to-back device fences in barrier helpers.
    header = header.replace(
        "atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);\n"
        "    threadgroup_barrier(mem_flags::mem_device);",
        "threadgroup_barrier(mem_flags::mem_device);",
    )

    # Scalar bool input 'active' in FilteredResidentSampler is passed as const bool* in CUDA.
    source = source.replace("if(!active)", "if(!(*active))")

    return CUDA_COMPAT_HEADER + "\n" + header + "\n", source

