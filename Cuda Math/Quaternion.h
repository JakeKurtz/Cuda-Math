#ifndef _CML_QUAT_
#define _CML_QUAT_

#include "CudaCommon.h"
#include "GLCommon.h"
#include "Vector.h"
#include "Matrix.h"

namespace cml
{
	template <ASX::ID, class T> struct quat_t;

	template <typename T> using quat = quat_t<ASX::ID_value, T>;

	typedef quat<float>		quatf;
	typedef quat<int32_t>	quati;
	typedef quat<uint32_t>	quatu;

	template <ASX::ID t_id, typename T>
	struct _ALIGN(sizeof(T) * 4) quat_t
	{
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T a; ASX_ASA dummy1; };
		union { T i; ASX_ASA dummy2; };
		union { T j; ASX_ASA dummy3; };
		union { T k; ASX_ASA dummy4; };

		_HOST_DEVICE quat_t() {};

		_HOST_DEVICE quat_t(T a, T i, T j, T k) : a(a), i(i), j(j), k(k) {};

		_HOST_DEVICE quat_t(T s) : a(s), i(s), j(s), k(s) {};

		template<ASX::ID u_id>
		_HOST_DEVICE quat_t(const quat_t<u_id, T>&q) : a(q.a), i(q.i), j(q.j), k(q.k) {};

		template<ASX::ID u_id>
		_HOST_DEVICE quat_t(T a, const vec3<u_id, T>&q) : a(a), i(q.i), j(q.j), k(q.k) {};

		_HOST_DEVICE quat_t(const float4 & q) : a(q.a), i(q.i), j(q.j), k(q.k) {};

		_HOST quat_t(const glm::quat_t& q) : a(q.w), i(q.x), j(q.y), k(q.z) {};

		template <ASX::ID u_id, class U>
		_HOST_DEVICE operator quat_t<u_id, U>() const
		{
			return quat_t<U>(a, i, j, k);
		};

		_HOST_DEVICE operator float4() const
		{
			return make_float4(a, i, j, k);
		};

		_HOST_DEVICE operator glm::quat_t() const
		{
			return glm::quat_t(a, i, j, k);
		};

		_HOST_DEVICE void print()
		{
			printf("%f + %fi + %fj + %fk\n", (float)a, (float)i, (float)j, (float)k);
		};

		template <ASX::ID u_id>
		_HOST_DEVICE void to_mat4(Matrix4x4<T>&mat)
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
				m00, m10, m20, 0,
				m01, m11, m21, 0,
				m02, m12, m22, 0,
				0,   0,   0,   1
				);
		}

		template <ASX::ID u_id>
		_HOST_DEVICE void angle_axis(T & angle, vec3<u_id, T>&axis)
		{
			if (vec3<ASX::ID_value, T>(i, j, k) == vec3<ASX::ID_value, T>(0.0)) {
				axis = vec3<ASX::ID_value, T>(1, 0, 0);
			}
			else {
				axis = normalize(vec3<ASX::ID_value, T>(i, j, k));
			}
			angle = 2.0 * ::atan2(::sqrt(i * i + j * j + k * k), a);
		}
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline bool operator==(const quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
	{
		return (q1.a == q2.a && q1.i == q2.i && q1.j == q2.j && q1.k == q2.k)
	}
	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline bool operator!=(const quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
	{
		return (q1.a != q2.a || q1.i != q2.i || q1.j != q2.j || q1.k != q2.k)
	}

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline quat_t<t_id, T> operator*=(const quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
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

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline auto operator*(const quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
		-> quat_t<ASX::ID_value, decltype(q1.a* q2.a)>
	{
		quat_t<ASX::ID_value, decltype(q1.a* q2.a)> temp(q2);
		return temp *= q2;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline quat_t<t_id, T> operator*=(quat_t<t_id, T>& q, const U s)
	{
		q.a *= s; q.i *= s; q.j *= s; q.k *= s;
		return q;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator*(const U s, const quat_t<t_id, T>& q)
		-> quat_t<ASX::ID_value, decltype(s* q.a)>
	{
		quat_t<ASX::ID_value, decltype(s * q.a)> temp(q);
		return temp *= s;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator*(const quat_t<t_id, T>& q, const U s)
		-> quat_t<ASX::ID_value, decltype(q.a* s)>
	{
		quat_t<ASX::ID_value, decltype(s * q.a)> temp(q);
		return temp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline quat_t<t_id, T> operator/=(quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
	{
		auto len = length(q2); auto len2 = len * len;
		auto recip = quat_t<U>(q2.a / len2, -q2.i / len2, -q2.j / len2, -q2.k / len2);
		return q1 *= recip;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline auto operator/(const quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
		-> quat_t<ASX::ID_value, decltype(q1.a/ q2.a)>
	{
		quat_t<ASX::ID_value, decltype(q1.a/ q2.a)> temp(q2);
		return temp /= q2;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline quat_t<t_id, T> operator/=(quat_t<t_id, T>& q, const U s)
	{
		q.a /= s; q.i /= s; q.j /= s; q.k /= s;
		return q;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator/(const U s, const quat_t<t_id, T>& q)
		-> quat_t<ASX::ID_value, decltype(s/ q.a)>
	{
		quat_t<ASX::ID_value, decltype(s / q.a)> temp(q);
		return temp /= s;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator/(const quat_t<t_id, T>& q, const U s)
		-> quat_t<ASX::ID_value, decltype(q.a/ s)>
	{
		quat_t<ASX::ID_value, decltype(s / q.a)> temp(q);
		return temp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
		inline quat_t<t_id, T> operator+=(quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
	{
		q1.a += q2.a; q1.i += q2.i; q1.j += q2.j; q1.k += q2.k;
		return q1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
		inline auto operator+(const quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
		-> quat_t<ASX::ID_value, decltype(q1.a + q2.a)>
	{
		quat_t<ASX::ID_value, decltype(q1.a + q2.a)> temp(q2);
		return temp += q2;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline quat_t<t_id, T> operator+=(quat_t<t_id, T>& q, const U s)
	{
		q.a += s; q.i += s; q.j += s; q.k += s;
		return q;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator+(const U s, const quat_t<t_id, T>& q)
		-> quat_t<ASX::ID_value, decltype(s + q.a)>
	{
		quat_t<ASX::ID_value, decltype(s + q.a)> temp(q);
		return temp += s;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator+(const quat_t<t_id, T>& q, const U s)
		-> quat_t<ASX::ID_value, decltype(q.a + s)>
	{
		quat_t<ASX::ID_value, decltype(s + q.a)> temp(q);
		return temp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline quat_t<t_id, T> operator-=(quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
	{
		q1.a -= q2.a; q1.i -= q2.i; q1.j -= q2.j; q1.k -= q2.k;
		return q1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > _HOST_DEVICE
	inline auto operator-(const quat_t<t_id, T>& q1, const quat_t<u_id, U>& q2)
		-> quat_t<ASX::ID_value, decltype(q1.a - q2.a)>
	{
		quat_t<ASX::ID_value, decltype(q1.a - q2.a)> temp(q2);
		return temp -= q2;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline quat_t<t_id, T> operator-=(quat_t<t_id, T>& q, const U s)
	{
		q.a -= s; q.i -= s; q.j -= s; q.k -= s;
		return q;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator-(const U s, const quat_t<t_id, T>& q)
		-> quat_t<ASX::ID_value, decltype(s - q.a)>
	{
		quat_t<ASX::ID_value, decltype(s - q.a)> temp(q);
		return temp -= s;
	};

	template <ASX::ID t_id, class T, class U> _HOST_DEVICE
	inline auto operator-(const quat_t<t_id, T>& q, const U s)
		-> quat_t<ASX::ID_value, decltype(q.a - s)>
	{
		quat_t<ASX::ID_value, decltype(s - q.a)> temp(q);
		return temp -= s;
	};


	template <ASX::ID t_id, class T> _HOST_DEVICE
	inline quat_t<ASX::ID_value, T> operator-(const quat_t<t_id, T>& q)
	{
		return quat_t<ASX::ID_value, T>(-q.a, -q.i, -q.j, -q.k);
	}

	template <ASX::ID t_id, class T> _HOST_DEVICE
	inline quat_t<ASX::ID_value, T> conjugate(const quat_t<t_id, T>& q)
	{
		return quat_t<ASX::ID_value, T>(q.a, -q.i, -q.j, -q.k);
	}

	template <ASX::ID t_id, class T> _HOST_DEVICE
	inline float length(const quat_t<t_id, T>& q)
	{
		return ::sqrt(q.i * q.i + q.j * q.j + q.k * q.k + q.a * q.a);
	}

	template <ASX::ID t_id, class T> _HOST_DEVICE
	inline quat_t<ASX::ID_value, T> normalize(const quat_t<t_id, T>& q)
	{
		quat_t<ASX::ID_value, T> temp(q);

		float l = length(temp);
		return (l == 0.f) ? temp : temp / l;
	}

}
#endif // _CML_QUAT_