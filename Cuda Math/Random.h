#ifndef _CML_RANDOM_
#define _CML_RANDOM_

#include "GLCommon.h"
#include "CudaCommon.h"

namespace cml
{
	_HOST_DEVICE uint32_t lowerbias32(uint32_t x);
	_HOST_DEVICE uint32_t rand();

	_HOST_DEVICE float rand_float();
	_HOST_DEVICE float rand_float(float min, float max);

	_HOST_DEVICE int rand_int();
	_HOST_DEVICE int rand_int(int min, int max);
}

#endif // _CML_RANDOM_