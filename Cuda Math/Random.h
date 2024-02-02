#ifndef _CML_RANDOM_
#define _CML_RANDOM_

#include "cuda_common.h"
#include "gl_common.h"

namespace cml
{
	CML_FUNC_DECL uint32_t lowerbias32(uint32_t x);
	CML_FUNC_DECL uint32_t rand();

	CML_FUNC_DECL float rand_float();
	CML_FUNC_DECL float rand_float(float min, float max);

	CML_FUNC_DECL int rand_int();
	CML_FUNC_DECL int rand_int(int min, int max);
}

#endif // _CML_RANDOM_