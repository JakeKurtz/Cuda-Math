/* ---------------------------------------------------------------------------------- *
*
*    MIT License
*
*    Copyright(c) 2024 Jake Kurtz
*
*    Permission is hereby granted, free of charge, to any person obtaining a copy
*    of this softwareand associated documentation files(the "Software"), to deal
*    in the Software without restriction, including without limitation the rights
*    to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
*    copies of the Software, and to permit persons to whom the Software is
*    furnished to do so, subject to the following conditions :
*
*    The above copyright noticeand this permission notice shall be included in all
*    copies or substantial portions of the Software.
*
*    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
*    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
*    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
*    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
*    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
*    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
*    SOFTWARE.
*
* ---------------------------------------------------------------------------------- */

#include "spectral.h"

namespace cml
{

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const spec_t<T>& s1, const spec_t<T>& s2)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			auto ss_1 == s1[i];
			auto ss_2 == s2[i];

			auto p_1 = ss_1.power; auto w_1 = ss_1.wavelength;
			auto p_2 = ss_2.power; auto w_2 = ss_2.wavelength;

			if (p_1 != p_2 || w_1 != w_2) return false;
		}
		return true;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const spec_t<T>& s1, const spec_t<T>& s2)
	{
		return !(s1 == s2);
	}

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator*=(spec_t<T>& s1, const spec_t<T>& s2)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s1[i].power *= s2[i].power;
		}
		return s1;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator*=(spec_t<T>& s, const T x)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s1[i].power *= x;
		}
		return s1;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator*(const spec_t<T>& s1, const spec_t<T>& s2)
	{
		spec_t<T> tmp(s1);
		return tmp *= s2;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator*(const T x, const spec_t<T>& s)
	{
		spec_t<T> tmp(s);
		return tmp *= x;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator*(const spec_t<T>& s, const T x)
	{
		spec_t<T> tmp(s);
		return tmp *= x;
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator/=(spec_t<T>& s1, const spec_t<T>& s2)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s1[i].power /= s2[i].power;
		}
		return s1;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator/=(spec_t<T>& s, const T x)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s[i].power /= x;
		}
		return s;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator/(const spec_t<T>& s1, const spec_t<T>& s2)
	{
		spec_t<T> tmp(s1);
		return tmp /= s2;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator/(const T x, const spec_t<T>& s)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s[i].power = x / s[i].power;
		}
		return s;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator/(const spec_t<T>& s, const T x)
	{
		spec_t<T> tmp(s1);
		return tmp /= x;
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator+=(spec_t<T>& s1, const spec_t<T>& s2)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s1[i].power += s2[i].power;
		}
		return s1;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator+=(spec_t<T>& s, const T x)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s[i].power += x;
		}
		return s;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator+(const spec_t<T>& s1, const spec_t<T>& s2)
	{
		spec_t<T> tmp(s1);
		return tmp += s2;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator+(const T x, const spec_t<T>& s)
	{
		spec_t<T> tmp(s);
		return tmp += x;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator+(const spec_t<T>& s, const T x)
	{
		spec_t<T> tmp(s);
		return tmp += x;
	}

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator-=(spec_t<T>& s1, const spec_t<T>& s2)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s1[i].power -= s2[i].power;
		}
		return s1;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T>& operator-=(spec_t<T>& s, const T x)
	{
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			s[i].power -= x;
		}
		return s;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator-(const spec_t<T>& s1, const spec_t<T>& s2)
	{
		spec_t<T> tmp(s1);
		return tmp -= s2;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator-(const T x, const spec_t<T>& s)
	{
		spec_t<T> tmp(-s);
		return tmp += x;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator-(const spec_t<T>& s, const T x)
	{
		spec_t<T> tmp(s);
		return tmp -= x;
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> operator-(const spec_t<T>& s)
	{
		spec_t<T> tmp(s);
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			tmp[i].power = -tmp[i].power;
		}
		return tmp;
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> floor(const spec_t<T>& s)
	{
		spec_t<T> tmp(s);
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			tmp[i].power = ::floor(tmp[i].power);
		}
		return tmp;
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> ceil(const spec_t<T>& s)
	{
		spec_t<T> tmp(s);
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			tmp[i].power = ::ceil(tmp[i].power);
		}
		return tmp;
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> frac(const spec_t<T>& s)
	{
		{
			spec_t<T> tmp(s);
			for (int i = 0; i < LAMBDA_SAMPLES; i++)
			{
				tmp[i].power = ::frac(tmp[i].power);
			}
			return tmp;
		}
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> clamp(const spec_t<T>& s, const T min, const T max)
	{
		spec_t<T> tmp(s);
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			tmp[i].power = clamp(tmp[i].power, min, max);
		}
		return tmp;
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR spec_t<T> pow(const spec_t<T>& s, const T p)
	{
		spec_t<T> tmp(s);
		for (int i = 0; i < LAMBDA_SAMPLES; i++)
		{
			tmp[i].power = pow(tmp[i].power, p);
		}
		return tmp;
	}

	template <class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> color_match(T wavelength)
	{
		vec3<T> xyz;
#if defined(CIE_FIT)
		xyz.x = CIE_X(wavelength);
		xyz.y = CIE_Y(wavelength);
		xyz.z = CIE_Z(wavelength);
#else
		auto i = (wavelength - CIE_MIN) / CIE_STEP;
		auto t = frac(i);

		int i0 = int(::floor(i)); int i1 = int(::ceil(i));

		float D65 = mix(CIE_D65[i0], CIE_D65[i1], t) / CIE_Y_INTEGRAL;

		xyz.x = mix(CIE_X[i0], CIE_X[i1], t) * D65;
		xyz.y = mix(CIE_Y[i0], CIE_Y[i1], t) * D65;
		xyz.z = mix(CIE_Z[i0], CIE_Z[i1], t) * D65;
#endif
		return xyz;
	}
}