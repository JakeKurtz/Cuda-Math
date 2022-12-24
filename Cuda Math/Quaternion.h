#ifndef _JEK_QUAT_
#define _JEK_QUAT_

#include "CudaCommon.h"
#include "GLCommon.h"
#include "Vector.h"
#include "Matrix.h"

namespace jek
{
	template <class T> struct _ALIGN(16) Quat
	{
		static_assert(sizeof(T) == 4, "Type must be 4 bytes");
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		T a{}, i{}, j{}, k{};
		_HOST_DEVICE Quat() {};
		_HOST_DEVICE Quat(T a, T i, T j, T k) : a(a), i(i), j(j), k(k) {};
		_HOST_DEVICE Quat(T s) : a(s), i(s), j(s), k(s) {};
		_HOST_DEVICE Quat(const Quat<T>&q) : a(q.a), i(q.i), j(q.j), k(q.k) {};
		_HOST_DEVICE Quat(T a, const Vec3<T>&q) : a(a), i(q.i), j(q.j), k(q.k) {};

		_HOST_DEVICE Quat(const float4& q) : a(q.a), i(q.i), j(q.j), k(q.k) {};
		_HOST Quat(const glm::quat & q) : a(q.w), i(q.x), j(q.y), k(q.z) {};

		template <class U>
		_HOST_DEVICE operator Quat<U>() const
		{
			return Quat<U>(a, i, j, k);
		};
		_HOST_DEVICE operator float4() const
		{
			return make_float4(a, i, j, k);
		};
		_HOST operator glm::quat() const
		{
			return glm::quat(a, i, j, k);
		};

		_HOST_DEVICE void print()
		{
			printf("%f + %fi + %fj + %fk\n", (float)a, (float)i, (float)j, (float)k);
		};

		_HOST_DEVICE void to_mat4(Matrix4x4<T>& mat)
		{
			auto len = length(*this);
			auto s = 2.f / (len * len);

			auto m00 = 1.f - s * (j * j + k * k);
			auto m01 = s * (i * j - k * a);
			auto m02 = s * (i * k + j * a);
			auto m10 = s * (i * j + k * a);
			auto m11 = 1.f - s * (i * i + k * k);
			auto m12 = s * (j * k - i * a);
			auto m20 = s * (i * k - j * a);
			auto m21 = s * (j * k + i * a);
			auto m22 = 1.f - s * (i * i + j * j);

			mat = Matrix4x4<T>(
				m00, m10, m20, 0.f,
				m01, m11, m21, 0.f,
				m02, m12, m22, 0.f,
				0.f, 0.f, 0.f, 1.f
			);
		}
		_HOST_DEVICE void angle_axis(T& angle, Vec3<T>& axis)
		{
			if (Vec3<T>(i, j, k) == Vec3<T>(0.0)) {
				axis = Vec3<T>(1, 0, 0);
			}
			else {
				axis = normalize(Vec3<T>(i, j, k));
			}
			angle = 2.0 * ::atan2(::sqrt(i * i + j * j + k * k), a);
		}
	};

	typedef Quat<float>		Quatf;
	typedef Quat<int32_t>	Quati;
	typedef Quat<uint32_t>	Quatu;

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline bool operator==(const Quat<T>& q1, const Quat<U>& q2)
	{
		return (q1.a == q2.a && q1.i == q2.i && q1.j == q2.j && q1.k == q2.k)
	}
	template <class T, class U> _HOST_DEVICE
		inline bool operator!=(const Quat<T>& q1, const Quat<U>& q2)
	{
		return (q1.a != q2.a || q1.i != q2.i || q1.j != q2.j || q1.k != q2.k)
	}

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a * q2.a)>
	{
		auto s1 = q1.a * q2.a;
		auto s2 = q1.i * q2.i;
		auto s3 = q1.j * q2.j;
		auto s4 = q1.k * q2.k;

		auto s5 = (q1.a + q1.i) * (q2.a + q2.i);
		auto s6 = (q1.j - q1.k) * (q2.j + q2.k);

		auto s7 = (q1.a + q1.j) * (q2.a + q2.j);
		auto s8 = (q1.i + q1.k) * (q2.i - q2.k);

		auto s9 = (q1.a + q1.k) * (q2.a + q2.k);
		auto s10 = (q1.i - q1.j) * (q2.i + q2.j);

		auto a = s1 - s2 - s3 - s4;
		auto i = (s5 - s1 - s2) + (s6 - s3 + s4);
		auto j = (s7 - s1 - s3) + (s8 - s2 + s4);
		auto k = (s9 - s1 - s4) + (s10 - s2 + s3);

		q1.a = a; q1.i = i; q1.j = j; q1.k = k;

		return q1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Quat<T>& q, const U s)
		-> Quat<decltype(q.x * s)>
	{
		q = { s * q.a, s * q.i, s * q.j, s * q.k };
		return q;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a * q2.a)>
	{
		auto temp(q1);
		return temp *= q2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const T s, const Quat<U>& q)
		-> Quat<decltype(s * q.a)>
	{
		auto temp(q);
		return temp *= s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Quat<T>& q, const U s)
		-> Quat<decltype(q.a * s)>
	{
		auto temp(q);
		return temp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */
	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a / q2.a)>
	{
		auto len = length(q2); auto len2 = len*len;
		auto recip = Quat<U>(q2.a / len2, -q2.i / len2, -q2.j / len2, -q2.k / len2);
		return q1 *= recip;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Quat<T>& q, const U s)
		-> Quat<decltype(q.a / s)>
	{
		q = {q.a / s, q.k / s, q.j / s, q.k / s};
		return q;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a* q2.a)>
	{
		auto temp(q1);
		return temp /= q2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const T s, const Quat<U>& q)
		-> Quat<decltype(s / q.a)>
	{
		Quat<decltype(s / q.a)>temp(s);
		return temp /= q;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Quat<T>& q, const U s)
		-> Quat<decltype(q.x / s)>
	{
		auto temp(q);
		return temp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a + q2.a)>
	{
		q1 = { q1.a + q2.a, q1.i + q2.i, q1.j + q2.j, q1.k + q2.k };
		return q1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Quat<T>& q, const U s)
		-> Quat<decltype(q.a + s)>
	{
		q = { s + q.a, s + q.i, s + q.j, s + q.k };
		return q;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a + q2.a)>
	{
		auto temp(q1);
		return temp += q2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const T s, const Quat<U>& q)
		-> Quat<decltype(s + q.a)>
	{
		auto temp(q);
		return temp += s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Quat<T>& q, const U s)
		-> Quat<decltype(q.a + s)>
	{
		auto temp(q);
		return temp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a - q2.a)>
	{
		q1 = { q1.a - q2.a, q1.i - q2.i, q1.j - q2.j, q1.k - q2.k };
		return q1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Quat<T>& q, const U s)
		-> Quat<decltype(q.a - s)>
	{
		q = { q.a - s, q.i - s, q.j - s, q.k - s };
		return q;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Quat<T>& q1, const Quat<U>& q2)
		-> Quat<decltype(q1.a - q2.a)>
	{
		auto temp(q1);
		return temp -= q2;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const T s, const Quat<U>& q)
		-> Quat<decltype(s - q.a)>
	{
		Quat<U>temp(s);
		return temp -= q;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Quat<T>& q, const U s)
		-> Quat<decltype(q.a - s)>
	{
		auto temp(q);
		return temp -= s;
	}

	template<class T> _HOST_DEVICE
		inline Quat<T> operator-(const Quat<T>& q)
	{
		return {-q.a, -q.i, -q.j, -q.k};
	}
	template<class T> _HOST_DEVICE
		inline Quat<T> conjugate(const Quat<T>& q)
	{
		return {q.a, -q.i, -q.j, -q.k};
	}
	template<class T> _HOST_DEVICE
		inline float length(const Quat<T>& q)
	{
		return ::sqrtf(q.i*q.i + q.j*q.j + q.k*q.k + q.a*q.a);
	}
	template <class T> _HOST_DEVICE
		inline Quat<T> normalize(const Quat<T>& q)
	{
		float l = length(q);
		return (l == 0.f) ? q : q / l;
	}

}

#endif // _JEK_QUAT_