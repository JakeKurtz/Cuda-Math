#ifndef _CML_SAMPLE_
#define _CML_SAMPLE_

#include "vec.h"

namespace cml
{
	CLM_FUNC_DECL vec2f uniform_sample_square();

	CLM_FUNC_DECL vec3f uniform_sample_hemisphere(const vec2f& u);
	CLM_FUNC_DECL vec3f uniform_sample_hemisphere();

	CLM_FUNC_DECL vec3f uniform_sample_sphere(const vec2f& u);
	CLM_FUNC_DECL vec3f uniform_sample_sphere();

	CLM_FUNC_DECL vec2f uniform_sample_disk(const vec2f& u);
	CLM_FUNC_DECL vec2f uniform_sample_disk();

	CLM_FUNC_DECL vec2f concentric_sample_disk(const vec2f& u);
	CLM_FUNC_DECL vec2f concentric_sample_disk();

	CLM_FUNC_DECL vec3f cosine_sample_hemisphere(const vec2f& u);
	CLM_FUNC_DECL vec3f cosine_sample_hemisphere();
}

#endif // _CML_SAMPLE_

