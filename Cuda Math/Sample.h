#ifndef _CML_SAMPLE_
#define _CML_SAMPLE_

#include "Vector.h"

namespace cml
{
	_HOST_DEVICE vec2f uniform_sample_square();

	_HOST_DEVICE vec3f uniform_sample_hemisphere(const vec2f& u);
	_HOST_DEVICE vec3f uniform_sample_hemisphere();

	_HOST_DEVICE vec3f uniform_sample_sphere(const vec2f& u);
	_HOST_DEVICE vec3f uniform_sample_sphere();

	_HOST_DEVICE vec2f uniform_sample_disk(const vec2f& u);
	_HOST_DEVICE vec2f uniform_sample_disk();

	_HOST_DEVICE vec2f concentric_sample_disk(const vec2f& u);
	_HOST_DEVICE vec2f concentric_sample_disk();

	_HOST_DEVICE vec3f cosine_sample_hemisphere(const vec2f& u);
	_HOST_DEVICE vec3f cosine_sample_hemisphere();
}

#endif // _CML_SAMPLE_

