#ifndef _CML_SAMPLE_
#define _CML_SAMPLE_

#include "vec.h"

namespace cml
{
	CML_FUNC_DECL vec2f uniform_sample_square();

	CML_FUNC_DECL vec3f uniform_sample_hemisphere(const vec2f& u);
	CML_FUNC_DECL vec3f uniform_sample_hemisphere();

	CML_FUNC_DECL vec3f uniform_sample_sphere(const vec2f& u);
	CML_FUNC_DECL vec3f uniform_sample_sphere();

	CML_FUNC_DECL vec2f uniform_sample_disk(const vec2f& u);
	CML_FUNC_DECL vec2f uniform_sample_disk();

	CML_FUNC_DECL vec2f concentric_sample_disk(const vec2f& u);
	CML_FUNC_DECL vec2f concentric_sample_disk();

	CML_FUNC_DECL vec3f cosine_sample_hemisphere(const vec2f& u);
	CML_FUNC_DECL vec3f cosine_sample_hemisphere();
}

#endif // _CML_SAMPLE_

