#pragma once


#if defined(_TEST_)
#define _HOST_DEVICE
#define _DEVICE
#define _HOST
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