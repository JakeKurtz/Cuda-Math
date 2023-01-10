#ifndef _CML_RANDOM_
#define _CML_RANDOM_

#include "GLCommon.h"
#include "CudaCommon.h"

namespace cml
{
	CLM_FUNC_DECL uint32_t lowerbias32(uint32_t x);
	CLM_FUNC_DECL uint32_t rand();

	CLM_FUNC_DECL float rand_float();
	CLM_FUNC_DECL float rand_float(float min, float max);

	CLM_FUNC_DECL int rand_int();
	CLM_FUNC_DECL int rand_int(int min, int max);
}

#endif // _CML_RANDOM_