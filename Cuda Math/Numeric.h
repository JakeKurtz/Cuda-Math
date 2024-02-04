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

#ifndef _CML_MATH_
#define _CML_MATH_

#include "cuda_common.h"

namespace cml
{
	CML_CONSTANT float K_EPSILON	= 1e-6;
	CML_CONSTANT float K_HUGE		= 1e32;
	CML_CONSTANT float M_E			= 2.71828182845904523536028747135266250;	// The base of natural logarithms (e)
	CML_CONSTANT float M_LOG2E		= 1.44269504088896340735992468100189214;	// The logarithm to base 2 of M_E (log2(e))
	CML_CONSTANT float M_LOG10E	    = 0.43429448190325182765112891891660508;	// The logarithm to base 10 of M_E (log10(e))
	CML_CONSTANT float M_LN2		= 0.69314718055994530941723212145817656;	// The natural logarithm of 2 (loge(2))
	CML_CONSTANT float M_LN10		= 2.30258509299404568401799145468436421;	// The natural logarithm of 10 (loge(10))
	CML_CONSTANT float M_PI		    = 3.14159265358979323846264338327950288;	// Pi, the ratio of a circle's circumference to its diameter.
	CML_CONSTANT float M_2PI		= 6.28318530717958647692528676655900576;	// 2pi	
	CML_CONSTANT float M_PI_2		= 1.57079632679489661923132169163975144;	// Pi divided by two (pi/2)
	CML_CONSTANT float M_PI_4		= 0.78539816339744830961566084581987572;	// Pi divided by four (pi/4)
	CML_CONSTANT float M_PI_180     = 0.01745329251994329576923690768488612;	// Pi divided by four (pi/180)
	CML_CONSTANT float M_1_PI		= 0.31830988618379067153776752674502872;	// The reciprocal of pi (1/pi)
	CML_CONSTANT float M_1_2PI		= 0.15915494309189533576888376337251436;	// The reciprocal of 2pi (1/2pi)
	CML_CONSTANT float M_1_4PI		= 0.07957747154594766788444188168625718;	// The reciprocal of 4pi (1/4pi)
	CML_CONSTANT float M_2_PI		= 0.63661977236758134307553505349005744;	// Two times the reciprocal of pi (2/pi)
	CML_CONSTANT float M_2_SQRTPI	= 1.12837916709551257389615890312154517;	// Two times the reciprocal of the square root of pi (2/sqrt(pi))
	CML_CONSTANT float M_SQRT2		= 1.41421356237309504880168872420969808;	// The square root of two (sqrt(2))
	CML_CONSTANT float M_SQRT1_2	= 0.70710678118654752440084436210484903;	// The reciprocal of the square root of two (1/sqrt(2))

    template<NUMERIC_TYPE(T)>
    GLM_FUNC_DECL GLM_CONSTEXPR const T& min(const T& a, const T& b)
    {
        return (b < a) ? b : a;
    }

    template<NUMERIC_TYPE(T)>
    GLM_FUNC_DECL GLM_CONSTEXPR const T& max(const T& a, const T& b)
    {
        return (b > a) ? b : a;
    }

    template<NUMERIC_TYPE(T)> 
    CML_FUNC_DECL CML_CONSTEXPR T remap(const T h1, const T l1, const T h2, const T l2, const T v)
    {
        return l2 + (v - l1) * (h2 - l2) / (h1 - l1);
    }

    template<NUMERIC_TYPE(T)> 
    CML_FUNC_DECL CML_CONSTEXPR T frac(const T v)
    {
        return v - ::floor(v);
    }

    template<NUMERIC_TYPE(T)> 
    CML_FUNC_DECL CML_CONSTEXPR T clamp(const T f, const T a, const T b)
    {
        return max(a, min(f, b));
    }

    template <NUMERIC_TYPE(T)> 
    CML_FUNC_DECL CML_CONSTEXPR T mix(const T a, const T b, const T t)
    {
        return a * (static_cast<T>(1) - t) + b * t;
    }

    template<NUMERIC_TYPE(T)> 
    CML_FUNC_DECL CML_CONSTEXPR T smooth_step(const T a, const T b, const T x)
    {
        T y = clamp((x - a) / (b - a), 0, 1);
        return (y * y * (static_cast<T>(3) - (static_cast<T>(2) * y)));
    }

    template<NUMERIC_TYPE(T)>
    CML_FUNC_DECL CML_CONSTEXPR T pow2(const T x)
    {
        return x * x;
    }

    template<NUMERIC_TYPE(T)>
    CML_FUNC_DECL CML_CONSTEXPR T gauss(const T x, const T a, const T b, const T c)
    {
        return a * exp(-pow2((x - b) / c));
    }

    template<NUMERIC_TYPE(T)>
    CML_FUNC_DECL CML_CONSTEXPR T sigmoidal(const T x, const T a, const T b, const T c, const T d)
    {
        return d + (a - d) / (1.0 + ::pow(x / c, b));
    }

    template<NUMERIC_TYPE(T)>
    CML_FUNC_DECL CML_CONSTEXPR T sigmoidal(const T x, const T a, const T b, const T c)
    {
        return a / (1.0 + ::exp(-b * (x - c)));
    }
}
#endif // _CML_MATH_