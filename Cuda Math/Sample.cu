#include "Sample.h"
#include "Math.h"

namespace jek
{
	Vec2f uniform_sample_square()
	{
		return Vec2f(rand_float(), rand_float());
	}

	Vec3f uniform_sample_hemisphere(const Vec2f& u)
	{
		float z = u.x;
		float r = ::sqrt(::fmax(0.f, 1.f - z * z));
		float phi = 2 * M_PI * u.y;
		return Vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}
	Vec3f uniform_sample_hemisphere()
	{
		Vec2f u = Vec2f(rand_float(), rand_float());

		float z = u.x;
		float r = ::sqrt(::fmax(0.f, 1.f - z * z));
		float phi = 2 * M_PI * u.y;
		return Vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}

	Vec3f uniform_sample_sphere(const Vec2f& u)
	{
		float z = 1.f - 2.f * u.x;
		float r = ::sqrt(::fmax(0.f, 1.f - z * z));
		float phi = 2.f * M_PI * u.y;
		return Vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}
	Vec3f uniform_sample_sphere()
	{
		Vec2f u = Vec2f(rand_float(), rand_float());

		float z = 1.f - 2.f * u.x;
		float r = ::sqrtf(::fmax(0.f, 1.f - z * z));
		float phi = 2.f * M_PI * u.y;
		return Vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}

	Vec2f uniform_sample_disk(const Vec2f& u)
	{
		float r = ::sqrt(u.x);
		float theta = 2.f * M_PI * u.y;
		return Vec2f(r * ::cos(theta), r * ::sin(theta));
	}
	Vec2f uniform_sample_disk()
	{
		Vec2f u = Vec2f(rand_float(), rand_float());

		float r = ::sqrt(u.x);
		float theta = 2.f * M_PI * u.y;
		return Vec2f(r * ::cos(theta), r * ::sin(theta));
	}

	Vec2f concentric_sample_disk(const Vec2f& u)
	{
		// Map uniform random numbers to [-1,1]^2
		Vec2f u_offset = 2.f * u - Vec2f(1, 1);

		// Handle degeneracy at the origin 
		if (u_offset.x == 0.f && u_offset.y == 0.f)
			return Vec2f(0, 0);

		// Apply concentric mapping to point
		float theta, r;
		if (::abs(u_offset.x) > ::abs(u_offset.y)) {
			r = u_offset.x;
			theta = M_PI_4 * (u_offset.y / u_offset.x);
		}
		else {
			r = u_offset.y;
			theta = M_PI_2 - M_PI_4 * (u_offset.x / u_offset.y);
		}
		return r * Vec2f(::cos(theta), ::sin(theta));
	}
	Vec2f concentric_sample_disk()
	{
		Vec2f u = Vec2f(rand_float(), rand_float());

		// Map uniform random numbers to
		Vec2f u_offset = 2.f * u - Vec2f(1, 1);

		// Handle degeneracy at the origin 
		if (u_offset.x == 0.f && u_offset.y == 0.f)
			return Vec2f(0, 0);

		// Apply concentric mapping to point
		float theta, r;
		if (::abs(u_offset.x) > ::abs(u_offset.y)) {
			r = u_offset.x;
			theta = M_PI_4 * (u_offset.y / u_offset.x);
		}
		else {
			r = u_offset.y;
			theta = M_PI_2 - M_PI_4 * (u_offset.x / u_offset.y);
		}
		return r * Vec2f(::cos(theta), ::sin(theta));
	}

	Vec3f cosine_sample_hemisphere(const Vec2f& u)
	{
		Vec2f d = concentric_sample_disk(u);
		float z = sqrt(::fmax(0.f, 1.f - d.x * d.x - d.y * d.y));
		return Vec3f(d.x, d.y, z);
	}
	Vec3f cosine_sample_hemisphere()
	{
		Vec2f u = Vec2f(rand_float(), rand_float());

		Vec2f d = concentric_sample_disk(u);
		float z = sqrt(::fmax(0.f, 1.f - d.x * d.x - d.y * d.y));
		return Vec3f(d.x, d.y, z);
	}
}