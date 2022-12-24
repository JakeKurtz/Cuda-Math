#ifndef _JEK_COMPLEX_
#define _JEK_COMPLEX_

#include "CudaCommon.h"
#include "GLCommon.h"
#include "Vector.h"

namespace jek
{
	template <class T> struct _ALIGN(8) Complex
	{
		static_assert(sizeof(T) == 4, "Type must be 4 bytes");
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		T r{}, i{};
		_HOST_DEVICE Complex() {};
		template <class U>
		_HOST_DEVICE Complex(U r, U i) : r(r), i(i) {};
		template <class U>
		_HOST_DEVICE Complex(U r) : r(r), i(0) {};
		template <class U>
		_HOST_DEVICE Complex(const Complex<U>& c) : r(c.r), i(c.i) {};
		template <class U>
		_HOST_DEVICE Complex(const Vec2<U>& c) : r(c.r), i(c.i) {};

		_HOST_DEVICE Complex(const float2& c) : r(c.r), i(c.i) {};
		_HOST Complex(const glm::vec2& c) : r(c.r), i(c.i) {};

		template <class U>
		_HOST_DEVICE operator Vec2<U>() const
		{
			return Vec2<U>(r, i);
		};
		_HOST_DEVICE operator float2() const
		{
			return make_float2(r, i);
		};
		_HOST operator glm::vec2() const
		{
			return glm::vec2(r, i);
		};

		_HOST_DEVICE void polar_form(T& radius, T& theta)
		{
			radius = modulus(this);
			theta = ::arctan(i/r);
		};
		_HOST_DEVICE void print()
		{
			printf("%f + i%f\n", (float)r, (float)i);
		};
	};

	typedef Complex<float>		Complexf;
	typedef Complex<int32_t>	Complexi;
	typedef Complex<uint32_t>	Complexu;

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline bool operator==(const Complex<T>& c1, const Complex<U>& c2)
	{
		return (c1.r == c2.r && c1.i == c2.i)
	}
	template <class T, class U> _HOST_DEVICE
		inline bool operator!=(const Complex<T>& c1, const Complex<U>& c2)
	{
		return (c1.r != c2.r || c1.i != c2.i)
	}

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r * c2.r)>
	{
		auto s1 = c1.r * c2.r;
		auto s2 = c1.i * c2.i;
		auto s3 = (c1.r + c1.i)*(c2.r + c2.i);
		c1 = { s1-s2, s3-s1-s2 };
		return c1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Complex<T>& c, const U s)
		-> Complex<decltype(c.r * s)>
	{
		c = { s * c.r, s * c.i };
		return c;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r * c2.r)>
	{
		auto temp(c1);
		return temp *= c2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const T s, const Complex<U>& c)
		-> Complex<decltype(s * c.r)>
	{
		auto temp(c);
		return temp *= s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Complex<T>& c, const U s)
		-> Complex<decltype(c.r* s)>
	{
		auto temp(c);
		return temp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r / c2.r)>
	{
		auto s1 = c1.r * c2.r;
		auto s2 = c1.i * c2.i;
		auto s3 = (c1.r - c1.i) * (c2.r + c2.i);
		auto denom = c2.r*c2.r + c2.i*c2.i;
		c1 = { (s1 + s2) / denom, (s3 - s1 + s2) / denom };
		return c1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Complex<T>& c, const U s)
		-> Complex<decltype(c.r / s)>
	{
		c = { c.r / s, c.y / s };
		return c;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r / c2.r)>
	{
		auto temp(c1);
		return temp /= c2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const T s, const Complex<U>& c)
		-> Complex<decltype(s / c.r)>
	{
		Complex<decltype(s / c.r)>temp(s);
		return temp /= c;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Complex<T>& c, const U s)
		-> Complex<decltype(c.r / s)>
	{
		auto temp(c);
		return temp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r + c2.r)>
	{
		c1 = { c1.r + c2.r, c1.i + c2.i };
		return c1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Complex<T>& c, const U s)
		-> Complex<decltype(c.r + s)>
	{
		c1.r += s; c1.i += s;
		return c;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r + c2.r)>
	{
		auto temp(c1);
		return temp += c2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const T s, const Complex<U>& c)
		-> Complex<decltype(s + c.r)>
	{
		auto temp(c);
		return temp += s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Complex<T>& c, const U s)
		-> Complex<decltype(c.r + s)>
	{
		auto temp(c);
		return temp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r - c2.r)>
	{
		c1 = { c1.r - c2.r, c1.i - c2.i };
		return c1;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Complex<T>& c, const U s)
		-> Complex<decltype(c.r - s)>
	{
		c = { c.r - s, c.i - s };
		return c;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Complex<T>& c1, const Complex<U>& c2)
		-> Complex<decltype(c1.r - c2.r)>
	{
		auto temp(c1);
		return temp -= c2;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const T s, const Complex<U>& c)
		-> Complex<decltype(s - c.r)>
	{
		Complex<decltype(s - c.r)>temp(s);
		return temp -= c;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Complex<T>& c, const U s)
		-> Complex<decltype(c.r - s)>
	{
		auto temp(c);
		return temp -= s;
	}

	template<class T> _HOST_DEVICE
		inline Complex<T> floor(const Complex<T>& c)
	{
		return Complex<T>(::floor(c.r), ::floor(c.i));
	}
	template<class T> _HOST_DEVICE
		inline Complex<T> ceil(const Complex<T>& c)
	{
		return Complex<T>(::ceil(c.r), ::ceil(c.i));
	}
	template<class T> _HOST_DEVICE
		inline Complex<T> abs(const Complex<T>& c)
	{
		return Complex<T>(::abs(c.r), ::abs(c.i));
	}
	template<class T> _HOST_DEVICE
		inline Complex<T> clamp(const Complex<T>& c, const Complex<T>& min, const Complex<T>& max)
	{
		return Complex<T>(clamp(c.r, min.r, max.r), clamp(c.i, min.i, max.i));
	}
	template<class T> _HOST_DEVICE
		inline Complex<T> clamp(const Complex<T>& c, const T min, const T max)
	{
		return Complex<T>(clamp(c.r, min, max), clamp(c.i, min, max));
	}

	template<class T, class U> _HOST_DEVICE
		inline Complex<T> pow(const Complex<T>& c, const U p)
	{
		auto rn = ::pow((U)modulus(c), p);
		auto theta = ::atan((U)c.i/ (U)c.r);
		return {static_cast<T>(rn*::cos(p*theta)), static_cast<T>(rn*::sin(p*theta))};
	}
	template <class T> _HOST_DEVICE
		inline Complex<T> conjugate(const Complex<T>& c)
	{
		return {c.r, -c.i};
	}
	template <class T> _HOST_DEVICE
		inline float modulus(const Complex<T>& c)
	{
		return ::sqrt(c.r*c.r + c.i*c.i);
	}

}

#endif // _JEK_COMPLEX_