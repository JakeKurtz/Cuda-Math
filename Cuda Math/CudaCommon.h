#pragma once

#define Numeric_Type(T) typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type

#if defined(_UNIT_TEST_)

#define _HOST_DEVICE
#define _DEVICE
#define _HOST

struct float4 
{
	float x, y, z, w;
};
struct float3
{
	float x, y, z;
};
struct float2
{
	float x, y;
};

#else

#include "cuda_runtime.h"

#define check_CUDA_Errors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

extern void check_cuda(cudaError_t result, char const* const func, const char* const file, int const line);

#endif

#if defined(__CUDACC__) // NVCC

#include "cuda_runtime.h"

#define _ALIGN(n) alignas(n)
#define _CONSTANT __constant__ const

#define CLM_FUNC_DECL  __host__ __device__ 
#define CLM_CONSTEXPR constexpr

#elif defined(__GNUC__) // GCC
#define _ALIGN(n) __attribute__((aligned(n)))
#define _CONSTANT const

#elif defined(_MSC_VER) // MSVC
#define _ALIGN(n) alignas(n)
#define _CONSTANT const

#define CLM_FUNC_DECL
#define CLM_CONSTEXPR constexpr

#else
#error ""
#endif
