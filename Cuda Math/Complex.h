#ifndef _CML_COMPLEX_
#define _CML_COMPLEX_

#include "CudaCommon.h"
#include "GLCommon.h"
#include "Vector.h"

namespace cml
{
	template <ASX::ID, class T> struct complex_t;

	template <typename T> using complex = complex_t<ASX::ID_value, T>;

	typedef complex<double>		dcomplex;
	typedef complex<float>		fcomplex;
	typedef complex<int32_t>	icomplex;
	typedef complex<uint32_t>	ucomplex;

	template <ASX::ID t_id, typename T>
	struct _ALIGN(sizeof(T) * 2) complex_t
	{
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T r; ASX_ASA dummy1; };
		union { T i; ASX_ASA dummy2; };

		CLM_FUNC_DECL CLM_CONSTEXPR complex_t() {};

		CLM_FUNC_DECL CLM_CONSTEXPR complex_t(T r, T i) : r(r), i(i) {};

		CLM_FUNC_DECL CLM_CONSTEXPR complex_t(T r) : r(r), i(0) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR complex_t(const complex_t<u_id, T>&c) : r(c.r), i(c.i) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR complex_t(const vec2_t<u_id, T>&c) : r(c.r), i(c.i) {};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR complex_t<t_id, T>& operator=(const complex_t<u_id, U>& other)
		{
			r = static_cast<T>(other.r); i = static_cast<T>(other.i);
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR complex_t<t_id, T>& operator=(const vec2_t<u_id, U>& other)
		{
			r = static_cast<T>(other.x); i = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator complex_t<u_id, T>() const
		{
			return complex_t<ASX::ID_value, U>(r, i);
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec2_t<u_id, T>() const
		{
			return vec2<ASX::ID_value, U>(r, i);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR complex_t(const float2& c) : r(c.r), i(c.i) {};

		CLM_FUNC_DECL CLM_CONSTEXPR complex_t(const glm::vec2& c) : r(c.r), i(c.i) {};

		CLM_FUNC_DECL CLM_CONSTEXPR operator float2() const
		{
			return make_float2(r, i);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(r, i);
		};

		template<Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR void polar_form(U & radius, U & theta)
		{
			radius = modulus(this);
			theta = ::arctan(i / r);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR void print()
		{
			printf("%f + i%f\n", (float)r, (float)i);
		};
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline bool operator==(const complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
	{
		return (c1.r == c2.r && c1.i == c2.i)
	}

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline bool operator!=(const complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
	{
		return (c1.r != c2.r || c1.i != c2.i)
	}

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<t_id, T>& operator*=(complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
	{
		auto s1 = c1.r * c2.r;
		auto s2 = c1.i * c2.i;
		auto s3 = (c1.r + c1.i) * (c2.r + c2.i);

		c1.r = s1 - s2; c1.i = s3 - s1 - s2;

		return c1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator*(const complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
		-> complex_t<ASX::ID_value, decltype(c1.r * c2.r)>
	{
		complex_t<ASX::ID_value, decltype(c1.r* c2.r)> temp(c1);
		return temp *= c2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<t_id, T>& operator*=(complex_t<t_id, T>& c, const U s)
	{
		c.r *= s; c.i *= s;
		return c;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator*(const U s, const complex_t<t_id, T>& c)
		-> complex_t<ASX::ID_value, decltype(s * c.r)>
	{
		complex_t<ASX::ID_value, decltype(s* c.r)> temp(c);
		return temp *= s;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator*(const complex_t<t_id, T>& c, const U s)
		-> complex_t<ASX::ID_value, decltype(s* c.r)>
	{
		complex_t<ASX::ID_value, decltype(s* c.r)> temp(c);
		return temp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t <t_id, T>& operator/=(complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
	{
		auto s1 = c1.r * c2.r;
		auto s2 = c1.i * c2.i;
		auto s3 = (c1.r - c1.i) * (c2.r + c2.i);
		auto denom = c2.r * c2.r + c2.i * c2.i;

		c1.r = (s1 + s2) / denom; c1.i = (s3 - s1 + s2) / denom;

		return c1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator/(const complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
		-> complex_t<ASX::ID_value, decltype(c1.r / c2.r)>
	{
		complex_t<ASX::ID_value, decltype(c1.r / c2.r)> temp(c1);
		return temp /= c2;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t <t_id, T>& operator/=(complex_t<t_id, T>& c, const U s)
	{
		c.r = c.r / s; c.i = c.i / s;
		return c;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator/(const U s, const complex_t<t_id, T>& c)
		-> complex_t<ASX::ID_value, decltype(s / c.r)>
	{
		complex_t<ASX::ID_value, decltype(s / c.r)>temp(s);
		return temp /= c;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
		inline auto operator/(const complex_t<t_id, T>& c, const U s)
		-> complex_t<ASX::ID_value, decltype(c.r / s)>
	{
		complex_t<ASX::ID_value, decltype(s / c.r)>temp(s);
		return temp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<t_id, T>& operator+=(complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
	{
		c1.r = c1.r + c2.r; c1.i = c1.i + c2.i;
		return c1;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<t_id, T>& operator+=(complex_t<t_id, T>& c, const U s)
	{
		c.r += s; c.i += s;
		return c;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator+(const complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
		-> complex_t<ASX::ID_value, decltype(c1.r + c2.r)>
	{
		auto temp(c1);
		return temp += c2;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator+(const U s, const complex_t<t_id, T>& c)
		-> complex_t<ASX::ID_value, decltype(s + c.r)>
	{
		complex_t<ASX::ID_value, decltype(s + c.r)>temp(s);
		return temp += s;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator+(const complex_t<t_id, T>& c, const U s)
		-> complex_t<ASX::ID_value, decltype(c.r + s)>
	{
		complex_t<ASX::ID_value, decltype(s + c.r)>temp(s);
		return temp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<t_id, T>& operator-=(complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
	{
		c1.r = c1.r - c2.r; c1.i = c1.i - c2.i;
		return c1;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<t_id, T>& operator-=(complex_t<t_id, T>& c, const U s)
	{
		c.r -= s; c.i -= s;
		return c;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator-(const complex_t<t_id, T>& c1, const complex_t<u_id, U>& c2)
		-> complex_t<ASX::ID_value, decltype(c1.r - c2.r)>
	{
		auto temp(c1);
		return temp -= c2;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator-(const U s, const complex_t<t_id, T>& c)
		-> complex_t<ASX::ID_value, decltype(s - c.r)>
	{
		complex_t<ASX::ID_value, decltype(s - c.r)>temp(s);
		return temp -= s;
	};

	template <ASX::ID t_id, class T, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	inline auto operator-(const complex_t<t_id, T>& c, const U s)
		-> complex_t<ASX::ID_value, decltype(c.r - s)>
	{
		complex_t<ASX::ID_value, decltype(s - c.r)>temp(s);
		return temp -= s;
	};


	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<ASX::ID_value, T> floor(const complex_t<t_id, T>& c)
	{
		return complex_t<ASX::ID_value, T>(::floor(c.r), ::floor(c.i));
	}

	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<ASX::ID_value, T> ceil(const complex_t<t_id, T>& c)
	{
		return complex_t<ASX::ID_value, T>(::ceil(c.r), ::ceil(c.i));
	}

	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<ASX::ID_value, T> abs(const complex_t<t_id, T>& c)
	{
		return complex_t<ASX::ID_value, T>(::abs(c.r), ::abs(c.i));
	}

	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<ASX::ID_value, T> clamp(const complex_t<t_id, T>& c, const complex_t<t_id, T>& min, const complex_t<t_id, T>& max)
	{
		return complex_t<ASX::ID_value, T>(clamp(c.r, min.r, max.r), clamp(c.i, min.i, max.i));
	}

	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<ASX::ID_value, T> clamp(const complex_t<t_id, T>& c, const T min, const T max)
	{
		return complex_t<ASX::ID_value, T>(clamp(c.r, min, max), clamp(c.i, min, max));
	}

	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<ASX::ID_value, T> pow(const complex_t<t_id, T>& c, const T p)
	{
		auto rn = ::pow(static_cast<T>(T)modulus(c), p);
		auto theta = ::atan(static_cast<T>(T)c.i / static_cast<T>(T)c.r);
		return complex_t<ASX::ID_value, T>(static_cast<T>(rn * ::cos(p * theta)), static_cast<T>(rn * ::sin(p * theta)));
	}

	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline complex_t<ASX::ID_value, T> conjugate(const complex_t<t_id, T>& c)
	{
		return complex_t<ASX::ID_value, T>(c.r, -c.i);
	}

	template<ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	inline float modulus(const complex_t<t_id, T>& c)
	{
		return ::sqrt(c.r * c.r + c.i * c.i);
	}
}

#endif // _CML_COMPLEX_