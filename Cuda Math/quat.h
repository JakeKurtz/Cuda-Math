#ifndef _CML_QUAT_
#define _CML_QUAT_

#include "cuda_common.h"
#include "gl_common.h"

#include "vec.h"
#include "mat.h"

namespace cml
{
	template <ASX::ID, class T> struct quat_t;

	template <typename T> using quat = quat_t<ASX::ID_value, T>;

	typedef quat<double>	quatd;
	typedef quat<float>		quatf;
	typedef quat<int32_t>	quati;
	typedef quat<uint32_t>	quatu;

	template <ASX::ID t_id, typename T>
	struct CML_ALIGN(sizeof(T) * 4) quat_t
	{
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T a; ASX_ASA dummy1; };
		union { T i; ASX_ASA dummy2; };
		union { T j; ASX_ASA dummy3; };
		union { T k; ASX_ASA dummy4; };

		CML_FUNC_DECL CML_CONSTEXPR quat_t() {};

		CML_FUNC_DECL CML_CONSTEXPR quat_t(T a, T i, T j, T k) : a(a), i(i), j(j), k(k) {};

		CML_FUNC_DECL CML_CONSTEXPR quat_t(T s) : a(s), i(s), j(s), k(s) {};

		template<ASX::ID u_id, class T>
		CML_FUNC_DECL CML_CONSTEXPR quat_t(const quat_t<u_id, T>&q) : a(q.a), i(q.i), j(q.j), k(q.k) {};

		template<ASX::ID u_id, class T>
		CML_FUNC_DECL CML_CONSTEXPR quat_t(T a, const vec3_t<u_id, T>&q) : a(a), i(q.i), j(q.j), k(q.k) {};

		CML_FUNC_DECL CML_CONSTEXPR quat_t(const float4 & q) : a(q.a), i(q.i), j(q.j), k(q.k) {};

		CML_FUNC_DECL CML_CONSTEXPR quat_t(const glm::quat& q) : a(q.w), i(q.x), j(q.y), k(q.z) {};

		template <ASX::ID u_id, class T>
		CML_FUNC_DECL CML_CONSTEXPR operator quat_t<u_id, T>() const
		{
			return quat<U>(a, i, j, k);
		};

		template <ASX::ID u_id>
		CML_FUNC_DECL CML_CONSTEXPR void angle_axis(T& angle, vec3_t<u_id, T>& axis)
		{
			if (vec3<T>(i, j, k)==vec3<T>(0)) {
				axis = vec3<T>(1, 0, 0);
			}
			else {
				axis = normalize(vec3<T>(i, j, k));
			}
			angle = 2.0*::atan2(::sqrt(i*i+j*j+k*k), a);
		}

		CML_FUNC_DECL CML_CONSTEXPR operator mat4x4<T>() const
		{
			auto len = length(*this);
			auto s = static_cast<T>(1)/(len*len);

			auto t00 = static_cast<T>(1)-s*(j*j+k*k);
			auto t01 = s*(i*j-k*a);
			auto t02 = s*(i*k+j*a);
			auto t10 = s*(i*j+k*a);
			auto t11 = static_cast<T>(1)-s*(i*i+k*k);
			auto t12 = s*(j*k-i*a);
			auto t20 = s*(i*k-j*a);
			auto t21 = s*(j*k+i*a);
			auto t22 = static_cast<T>(1)-s*(i*i+j*j);

			return mat4x4<T>(
				t00, t10, t20, 0,
				t01, t11, t21, 0,
				t02, t12, t22, 0,
				0, 0, 0, 1
				);
		}

		CML_FUNC_DECL CML_CONSTEXPR operator float4() const
		{
			return make_float4(a, i, j, k);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::quat_t() const
		{
			return glm::quat(a, i, j, k);
		};

		CML_FUNC_DECL CML_CONSTEXPR void print()
		{
			printf("%f + %fi + %fj + %fk\n", (float)a, (float)i, (float)j, (float)k);
		};

	};

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator*=(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator*=(quat_t<t_id, T>& q, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator*(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator*(const T s, const quat_t<t_id, T>& q);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator*(const quat_t<t_id, T>& q, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator/=(quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator/=(quat_t<t_id, T>& q, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator/(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator/(const T s, const quat_t<t_id, T>& q);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator/(const quat_t<t_id, T>& q, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator+=(quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator+=(quat_t<t_id, T>& q, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator+(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator+(const T s, const quat_t<t_id, T>& q);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator+(const quat_t<t_id, T>& q, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator-=(quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator-=(quat_t<t_id, T>& q, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const T s, const quat_t<t_id, T>& q);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const quat_t<t_id, T>& q, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const quat_t<t_id, T>& q);

	/* -------------------------------------------------------------------------- */
	/*                        Conjugate, Length, Normalize                        */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> conjugate(const quat_t<t_id, T>& q);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR float length(const quat_t<t_id, T>& q);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> normalize(const quat_t<t_id, T>& q);

}// namespace cml

#include "quat.inl"

#endif // _CML_QUAT_