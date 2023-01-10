#include "Sample.h"
#include "Numeric.h"

namespace cml
{
	vec2f uniform_sample_square()
	{
		return vec2f(rand_float(), rand_float());
	}

	vec3f uniform_sample_hemisphere(const vec2f& u)
	{
		float z = u.x;
		float r = ::sqrt(::fmax(0.f, 1.f - z * z));
		float phi = 2 * M_PI * u.y;
		return vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}
	vec3f uniform_sample_hemisphere()
	{
		vec2f u = vec2f(rand_float(), rand_float());

		float z = u.x;
		float r = ::sqrt(::fmax(0.f, 1.f - z * z));
		float phi = M_2PI * u.y;
		return vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}

	vec3f uniform_sample_sphere(const vec2f& u)
	{
		float z = 1.f - 2.f * u.x;
		float r = ::sqrt(::fmax(0.f, 1.f - z * z));
		float phi = M_2PI * u.y;
		return vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}
	vec3f uniform_sample_sphere()
	{
		vec2f u = vec2f(rand_float(), rand_float());

		float z = 1.f - 2.f * u.x;
		float r = ::sqrtf(::fmax(0.f, 1.f - z * z));
		float phi = M_2PI * u.y;
		return vec3f(r * ::cos(phi), r * ::sin(phi), z);
	}

	vec2f uniform_sample_disk(const vec2f& u)
	{
		float r = ::sqrt(u.x);
		float theta = M_2PI * u.y;
		return vec2f(r * ::cos(theta), r * ::sin(theta));
	}
	vec2f uniform_sample_disk()
	{
		vec2f u = vec2f(rand_float(), rand_float());

		float r = ::sqrt(u.x);
		float theta = M_2PI * u.y;
		return vec2f(r * ::cos(theta), r * ::sin(theta));
	}

	vec2f concentric_sample_disk(const vec2f& u)
	{
		// Map uniform random numbers to [-1,1]^2
		vec2f u_offset = 2.f * u - vec2f(1, 1);

		// Handle degeneracy at the origin 
		if (u_offset.x == 0.f && u_offset.y == 0.f)
			return vec2f(0, 0);

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
		return r * vec2f(::cos(theta), ::sin(theta));
	}
	vec2f concentric_sample_disk()
	{
		vec2f u = vec2f(rand_float(), rand_float());

		// Map uniform random numbers to
		vec2f u_offset = 2.f * u - vec2f(1, 1);

		// Handle degeneracy at the origin 
		if (u_offset.x == 0.f && u_offset.y == 0.f)
			return vec2f(0, 0);

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
		return r * vec2f(::cos(theta), ::sin(theta));
	}

	vec3f cosine_sample_hemisphere(const vec2f& u)
	{
		vec2f d = concentric_sample_disk(u);
		float z = sqrt(::fmax(0.f, 1.f - d.x * d.x - d.y * d.y));
		return vec3f(d.x, d.y, z);
	}
	vec3f cosine_sample_hemisphere()
	{
		vec2f u = vec2f(rand_float(), rand_float());

		vec2f d = concentric_sample_disk(u);
		float z = sqrt(::fmax(0.f, 1.f - d.x * d.x - d.y * d.y));
		return vec3f(d.x, d.y, z);
	}
}