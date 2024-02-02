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

#include "cmplx.h"

namespace cml
{
	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		return (c1.r == c2.r && c1.i == c2.i);
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		return (c1.r != c2.r || c1.i != c2.i);
	}

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator*=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		auto s1 = c1.r * c2.r;
		auto s2 = c1.i * c2.i;
		auto s3 = (c1.r + c1.i) * (c2.r + c2.i);

		c1.r = s1 - s2; c1.i = s3 - s1 - s2;

		return c1;
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator*=(cmplx_t<t_id, T>& c, const T s)
	{
		c.r *= s; c.i *= s;
		return c;
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator*(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		cmplx<T> temp(c1);
		return temp *= c2;
	}

	template <ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator*(const T s, const cmplx_t<t_id, T>& c)
	{
		cmplx<T> temp(c);
		return temp *= s;
	}

	template <ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator*(const cmplx_t<t_id, T>& c, const T s)
	{
		cmplx<T> temp(c);
		return temp *= s;
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t <t_id, T>& operator/=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		auto s1 = c1.r * c2.r;
		auto s2 = c1.i * c2.i;
		auto s3 = (c1.r + c1.i) * (c2.r - c2.i);
		auto denom = c2.r * c2.r + c2.i * c2.i;

		c1.r = (s1 + s2) / denom; c1.i = (s3 - s1 + s2) / denom;

		return c1;
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t <t_id, T>& operator/=(cmplx_t<t_id, T>& c, const T s)
	{
		c.r /= s; c.i /= s;
		return c;
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator/(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		cmplx<T> temp(c1);
		return temp /= c2;
	}

	template <ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator/(const T s, const cmplx_t<t_id, T>& c)
	{
		cmplx<T> temp(s, 0.0);
		return temp /= c;
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator/(const cmplx_t<t_id, T>& c, const T s)
	{
		cmplx<T> temp(c);
		return temp /= s;
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator+=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		c1.r += c2.r; c1.i += c2.i;
		return c1;
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator+=(cmplx_t<t_id, T>& c, const T s)
	{
		c.r += s;
		return c;
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator+(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		cmplx<T> temp(c1);
		return temp += c2;
	}

	template <ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator+(const T s, const cmplx_t<t_id, T>& c)
	{
		cmplx<T> temp(s);
		return temp += c;
	}

	template <ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator+(const cmplx_t<t_id, T>& c, const T s)
	{
		cmplx<T> temp(s);
		return temp += c;
	}

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator-=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		c1.r -= c2.r; c1.i -= c2.i;
		return c1;
	}

	template <ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator-=(cmplx_t<t_id, T>& c, const T s)
	{
		c.r -= s;
		return c;
	}

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2)
	{
		cmplx<T> temp(c1);
		return temp -= c2;
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const T s, const cmplx_t<t_id, T>& c)
	{
		cmplx<T> temp(s);
		return temp -= c;
	}

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const cmplx_t<t_id, T>& c, const T s)
	{
		cmplx<T> temp(c);
		return temp -= s;
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const cmplx_t<t_id, T>& c)
	{
		return cmplx<T>(-c.r, -c.i);
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> floor(const cmplx_t<t_id, T>& c)
	{
		return cmplx<T>(::floor(c.r), ::floor(c.i));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> ceil(const cmplx_t<t_id, T>& c)
	{
		return cmplx<T>(::ceil(c.r), ::ceil(c.i));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> frac(const cmplx_t<t_id, T>& c)
	{
		return cmplx<T>(frac(c.r), frac(c.i));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Abs                                    */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> abs(const cmplx_t<t_id, T>& c)
	{
		return cmplx<T>(::abs(c.r), ::abs(c.i));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> clamp(const cmplx_t<t_id, T>& c, const cmplx_t<u_id, T>& min, const cmplx_t<v_id, T>& max)
	{
		return cmplx<T>(clamp(c.r, min.r, max.r), clamp(c.i, min.i, max.i));
	}

	template<ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> clamp(const cmplx_t<t_id, T>& c, const T min, const T max)
	{
		return cmplx<T>(clamp(c.r, min, max), clamp(c.i, min, max));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> pow(const cmplx_t<t_id, T>& c, const T p)
	{
		auto rn = ::pow(modulus(c), p);
		auto theta = ::atan2(c.i, c.r) * p;
		return cmplx<T>(rn * ::cos(theta), rn * ::sin(theta));
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Conjugate                                 */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> conjugate(const cmplx_t<t_id, T>& c)
	{
		return cmplx<T>(c.r, -c.i);
	}

	/* -------------------------------------------------------------------------- */
	/*                                   Modulus                                  */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T> 
	CML_FUNC_DECL CML_CONSTEXPR float modulus(const cmplx_t<t_id, T>& c)
	{
		return ::sqrt(c.r * c.r + c.i * c.i);
	}
} // namespace cml