#ifndef _JEK_SAMPLE_
#define _JEK_SAMPLE_

#include "Vector.h"

namespace jek
{
	_HOST_DEVICE Vec2f uniform_sample_square();

	_HOST_DEVICE Vec3f uniform_sample_hemisphere(const Vec2f& u);
	_HOST_DEVICE Vec3f uniform_sample_hemisphere();

	_HOST_DEVICE Vec3f uniform_sample_sphere(const Vec2f& u);
	_HOST_DEVICE Vec3f uniform_sample_sphere();

	_HOST_DEVICE Vec2f uniform_sample_disk(const Vec2f& u);
	_HOST_DEVICE Vec2f uniform_sample_disk();

	_HOST_DEVICE Vec2f concentric_sample_disk(const Vec2f& u);
	_HOST_DEVICE Vec2f concentric_sample_disk();

	_HOST_DEVICE Vec3f cosine_sample_hemisphere(const Vec2f& u);
	_HOST_DEVICE Vec3f cosine_sample_hemisphere();
}

#endif // _JEK_SAMPLE_

