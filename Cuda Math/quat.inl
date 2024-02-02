#include "quat.h"

namespace cml
{
	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		return (q1.a==q2.a&&q1.i==q2.i&&q1.j==q2.j&&q1.k==q2.k)
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		return (q1.a!=q2.a||q1.i!=q2.i||q1.j!=q2.j||q1.k!=q2.k)
	}

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator*=(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		auto s1 = q1.a*q2.a;
		auto s2 = q1.i*q2.i;
		auto s3 = q1.j*q2.j;
		auto s4 = q1.k*q2.k;

		auto s5 = (q1.a+q1.i)*(q2.a+q2.i);
		auto s6 = (q1.j-q1.k)*(q2.j+q2.k);

		auto s7 = (q1.a+q1.j)*(q2.a+q2.j);
		auto s8 = (q1.i+q1.k)*(q2.i-q2.k);

		auto s9 = (q1.a+q1.k)*(q2.a+q2.k);
		auto s10 = (q1.i-q1.j)*(q2.i+q2.j);

		auto a = s1-s2-s3-s4;
		auto i = (s5-s1-s2)+(s6-s3+s4);
		auto j = (s7-s1-s3)+(s8-s2+s4);
		auto k = (s9-s1-s4)+(s10-s2+s3);

		q1.a = a; q1.i = i; q1.j = j; q1.k = k;

		return q1;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator*=(quat_t<t_id, T>& q, const T s)
	{
		q.a *= s; q.i *= s; q.j *= s; q.k *= s;
		return q;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator*(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		quat<T> tmp(q1);
		return tmp *= q2;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator*(const T s, const quat_t<t_id, T>& q)
	{
		quat<T> tmp(q);
		return tmp *= s;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator*(const quat_t<t_id, T>& q, const T s)
	{
		quat<T> tmp(q);
		return tmp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator/=(quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		auto len = length(q2); auto len2 = len*len;
		quat<T> recip(q2.a/len2, -q2.i/len2, -q2.j/len2, -q2.k/len2);
		return q1 *= recip;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator/=(quat_t<t_id, T>& q, const T s)
	{
		q.a /= s; q.i /= s; q.j /= s; q.k /= s;
		return q;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator/(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		quat<T> tmp(q1);
		return tmp /= q2;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator/(const T s, const quat_t<t_id, T>& q)
	{
		quat<T> tmp(q);
		tmp.a = s/q.a; tmp.i = s/q.i; tmp.j = s/q.j; tmp.k = s/q.k;
		return tmp;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator/(const quat_t<t_id, T>& q, const T s)
	{
		quat<T> tmp(q);
		return tmp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator+=(quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		q1.a += q2.a; q1.i += q2.i; q1.j += q2.j; q1.k += q2.k;
		return q1;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator+=(quat_t<t_id, T>& q, const T s)
	{
		q.a += s; q.i += s; q.j += s; q.k += s;
		return q;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator+(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		quat<T> temp(q1);
		return temp += q2;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator+(const T s, const quat_t<t_id, T>& q)
	{
		quat<T> temp(q);
		return temp += s;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator+(const quat_t<t_id, T>& q, const T s)
	{
		quat<T> temp(q);
		return temp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator-=(quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		q1.a -= q2.a; q1.i -= q2.i; q1.j -= q2.j; q1.k -= q2.k;
		return q1;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat_t<t_id, T> operator-=(quat_t<t_id, T>& q, const T s)
	{
		q.a -= s; q.i -= s; q.j -= s; q.k -= s;
		return q;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const quat_t<t_id, T>& q1, const quat_t<u_id, T>& q2)
	{
		quat<T> temp(q1);
		return temp -= q2;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const T s, const quat_t<t_id, T>& q)
	{
		quat<T> temp(-q);
		return temp += s;
	};

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const quat_t<t_id, T>& q, const T s)
	{
		quat<T> temp(q);
		return temp -= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> operator-(const quat_t<t_id, T>& q)
	{
		return quat<T>(-q.a, -q.i, -q.j, -q.k);
	}

	/* -------------------------------------------------------------------------- */
	/*                        Conjugate, Length, Normalize                        */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> conjugate(const quat_t<t_id, T>& q)
	{
		return quat<T>(q.a, -q.i, -q.j, -q.k);
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR float length(const quat_t<t_id, T>& q)
	{
		return ::sqrt(q.i*q.i+q.j*q.j+q.k*q.k+q.a*q.a);
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR quat<T> normalize(const quat_t<t_id, T>& q)
	{
		quat<T> temp(q);

		float l = length(temp);
		return (l==0.f) ? temp : temp/l;
	}

} // namespace cml