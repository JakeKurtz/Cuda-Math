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

#ifndef _CML_CMPLX_
#define _CML_CMPLX_

#include "cuda_common.h"
#include "gl_common.h"
#include "vec.h"

namespace cml
{
	template <ASX::ID, typename T> struct cmplx_t;

	template <typename T> using cmplx = cmplx_t<ASX::ID_value, T>;

	typedef cmplx<double>	cmplxd;
	typedef cmplx<float>	cmplxf;
	typedef cmplx<int32_t>	cmplxi;
	typedef cmplx<uint32_t>	cmplxu;

	template <ASX::ID t_id, typename T>
	struct CML_ALIGN(sizeof(T) * 2) cmplx_t
	{
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T r; ASX_ASA dummy1; };
		union { T i; ASX_ASA dummy2; };

		CML_FUNC_DECL CML_CONSTEXPR cmplx_t() : r(0), i(0) {};

		CML_FUNC_DECL CML_CONSTEXPR cmplx_t(T r, T i) : r(r), i(i) {};

		CML_FUNC_DECL CML_CONSTEXPR cmplx_t(T r) : r(r), i(0) {};

		template<ASX::ID u_id>
		CML_FUNC_DECL CML_CONSTEXPR cmplx_t(const cmplx_t<u_id, T>&c) : r(c.r), i(c.i) {};

		template<ASX::ID u_id>
		CML_FUNC_DECL CML_CONSTEXPR cmplx_t(const vec2_t<u_id, T>&c) : r(c.r), i(c.i) {};

		template<ASX::ID u_id>
		CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator=(const cmplx_t<u_id, T>&other)
		{
			r = static_cast<T>(other.r); i = static_cast<T>(other.i);
			return *this;
		};

		template<ASX::ID u_id>
		CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator=(const vec2_t<u_id, T>&other)
		{
			r = static_cast<T>(other.x); i = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id>
		CML_FUNC_DECL CML_CONSTEXPR operator cmplx_t<u_id, T>() const
		{
			return cmplx<T>(r, i);
		};

		template<ASX::ID u_id>
		CML_FUNC_DECL CML_CONSTEXPR operator vec2_t<u_id, T>() const
		{
			return vec2<T>(r, i);
		};

		CML_FUNC_DECL CML_CONSTEXPR cmplx_t(const float2 & c) : r(c.x), i(c.y) {};

		CML_FUNC_DECL CML_CONSTEXPR cmplx_t(const glm::vec2 & c) : r(c.x), i(c.y) {};

		CML_FUNC_DECL CML_CONSTEXPR operator float2() const
		{
			return make_float2(r, i);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(r, i);
		};

		CML_FUNC_DECL CML_CONSTEXPR void polar_form(T & radius, T & theta)
		{
			radius = modulus(this);
			theta = ::atan(i / r);
		};

		CML_FUNC_DECL CML_CONSTEXPR void print()
		{
			printf("%f + i%f\n", (float)r, (float)i);
		};

		CML_FUNC_DECL CML_CONSTEXPR bool isnan() const
		{
			// TODO
		};
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator*=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator*=(cmplx_t<t_id, T>& c, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator*(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T >
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator*(const T s, const cmplx_t<t_id, T>& c);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator*(const cmplx_t<t_id, T>& c, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t <t_id, T>& operator/=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t <t_id, T>& operator/=(cmplx_t<t_id, T>& c, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator/(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator/(const T s, const cmplx_t<t_id, T>& c);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator/(const cmplx_t<t_id, T>& c, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator+=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator+=(cmplx_t<t_id, T>& c, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator+(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator+(const T s, const cmplx_t<t_id, T>& c);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator+(const cmplx_t<t_id, T>& c, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator-=(cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx_t<t_id, T>& operator-=(cmplx_t<t_id, T>& c, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const cmplx_t<t_id, T>& c1, const cmplx_t<u_id, T>& c2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const T s, const cmplx_t<t_id, T>& c);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const cmplx_t<t_id, T>& c, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> operator-(const cmplx_t<t_id, T>& c);

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> floor(const cmplx_t<t_id, T>& c);

	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> ceil(const cmplx_t<t_id, T>& c);

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> frac(const cmplx_t<t_id, T>& c);

	/* -------------------------------------------------------------------------- */
	/*                                     Abs                                    */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> abs(const cmplx_t<t_id, T>& c);

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> clamp(const cmplx_t<t_id, T>& c, const cmplx_t<u_id, T>& min, const cmplx_t<v_id, T>& max);

	template<ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> clamp(const cmplx_t<t_id, T>& c, const T min, const T max);

	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> pow(const cmplx_t<t_id, T>& c, const T p);

	/* -------------------------------------------------------------------------- */
	/*                                  Conjugate                                 */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR cmplx<T> conjugate(const cmplx_t<t_id, T>& c);

	/* -------------------------------------------------------------------------- */
	/*                                   Modulus                                  */
	/* -------------------------------------------------------------------------- */

	template<ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR float modulus(const cmplx_t<t_id, T>& c);
} // namespace cml

#include "cmplx.inl"

#endif // _CML_CMPLX_