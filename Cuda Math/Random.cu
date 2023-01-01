#include "Random.h"

namespace cml
{
	uint32_t lowerbias32(uint32_t x)
	{
		x ^= x >> 16;
		x *= 0xa812d533;
		x ^= x >> 15;
		x *= 0xb278e4ad;
		x ^= x >> 17;
		return x;
	}

	// Note:	host rand range:	[0, 32767)
	//			device rand range:	[0, 4294967296)

	uint32_t rand()
	{
#if defined(__CUDA_ARCH__)
		uint32_t i = threadIdx.x + blockIdx.x * blockDim.x;
		uint32_t j = threadIdx.y + blockIdx.y * blockDim.y;

		uint32_t seed = (i * 256 + j);
		return lowerbias32(seed);
#else
		return ::rand();
#endif
	}

	float rand_float()
	{
#if defined(__CUDA_ARCH__)
		return rand() * 0.00000000023283064365386962890625;
#else
		return rand() * 0.00003051850947599719229712820825;
#endif
	}
	float rand_float(float min, float max)
	{
		return ((rand_float() * (max - min)) + min);
	}

	int rand_int()
	{
		return rand();
	}
	int rand_int(int min, int max)
	{
		return ((rand_float() * (max - min)) + min);
	}
}