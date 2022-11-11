#pragma once


#if defined(_TEST_)
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

float4 make_float4(float x, float y, float z, float w)
{
	return float4();
}
float3 make_float3(float x, float y, float z)
{
	return float3();
}
float2 make_float2(float x, float y)
{
	return float2();
}

#else
#include "cuda_runtime.h"
#define _HOST_DEVICE __host__ __device__
#define _DEVICE __device__
#define _HOST __host__
#endif

#if defined(__CUDACC__) // NVCC
#define _ALIGN(n) __align__(n)
#define _CONSTANT __constant__ const
#elif defined(__GNUC__) // GCC
#define _ALIGN(n) __attribute__((aligned(n)))
#define _CONSTANT const
#elif defined(_MSC_VER) // MSVC
#define _ALIGN(n) __declspec(align(n))
#define _CONSTANT const
#else
#error ""
#endif