/* ---------------------------------------------------------------------------------- *
*
*    MIT License
*
*    Copyright(c) 2024 Jake Kurtz
*
*    Permission is hereby granted, free of charge, to any person obtaining a copy
*    of this softwareand associated documentation files(the "Software"), to deal
*    in the Software without restriction, including without limitation the rights
*    to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
*    copies of the Software, and to permit persons to whom the Software is
*    furnished to do so, subject to the following conditions :
*
*    The above copyright noticeand this permission notice shall be included in all
*    copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*    SOFTWARE.
*
* ---------------------------------------------------------------------------------- */

#pragma once

#define CML_DISABLE		0
#define CML_ENABLE		1

#define CML_ALIGNED_GENTYPES CML_ENABLE

#if CML_ALIGNED_GENTYPES == CML_ENABLE
	#define CML_ALIGN(n) alignas(n)
#else
	#define CML_ALIGN(n) alignas(0)
#endif

#define NUMERIC_TYPE(T) typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type

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

#define CML_CONSTANT __constant__ const

#define CML_FUNC_DECL  __host__ __device__ 
#define CML_CONSTEXPR constexpr

#elif defined(__GNUC__) // GCC
#define CML_ALIGN(n) __attribute__((aligned(n)))
#define CML_CONSTANT const

#elif defined(_MSC_VER) // MSVC
#define CML_CONSTANT const

#define CML_FUNC_DECL
#define CML_CONSTEXPR constexpr

#else
#error ""
#endif
