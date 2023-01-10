#ifndef _CML_MATH_
#define _CML_MATH_

#include "CudaCommon.h"

namespace cml
{
	_CONSTANT float K_EPSILON	= 1e-6;
	_CONSTANT float K_HUGE		= 1e32;
	_CONSTANT float M_E			= 2.71828182845904523536028747135266250;	// The base of natural logarithms (e)
	_CONSTANT float M_LOG2E		= 1.44269504088896340735992468100189214;	// The logarithm to base 2 of M_E (log2(e))
	_CONSTANT float M_LOG10E	= 0.43429448190325182765112891891660508;	// The logarithm to base 10 of M_E (log10(e))
	_CONSTANT float M_LN2		= 0.69314718055994530941723212145817656;	// The natural logarithm of 2 (loge(2))
	_CONSTANT float M_LN10		= 2.30258509299404568401799145468436421;	// The natural logarithm of 10 (loge(10))
	_CONSTANT float M_PI		= 3.14159265358979323846264338327950288;	// Pi, the ratio of a circle's circumference to its diameter.
	_CONSTANT float M_2PI		= 6.28318530717958647692528676655900576;	// 2pi	
	_CONSTANT float M_PI_2		= 1.57079632679489661923132169163975144;	// Pi divided by two (pi/2)
	_CONSTANT float M_PI_4		= 0.78539816339744830961566084581987572;	// Pi divided by four (pi/4)
	_CONSTANT float M_PI_180    = 0.01745329251994329576923690768488612;	// Pi divided by four (pi/180)
	_CONSTANT float M_1_PI		= 0.31830988618379067153776752674502872;	// The reciprocal of pi (1/pi)
	_CONSTANT float M_1_2PI		= 0.15915494309189533576888376337251436;	// The reciprocal of 2pi (1/2pi)
	_CONSTANT float M_1_4PI		= 0.07957747154594766788444188168625718;	// The reciprocal of 4pi (1/4pi)
	_CONSTANT float M_2_PI		= 0.63661977236758134307553505349005744;	// Two times the reciprocal of pi (2/pi)
	_CONSTANT float M_2_SQRTPI	= 1.12837916709551257389615890312154517;	// Two times the reciprocal of the square root of pi (2/sqrt(pi))
	_CONSTANT float M_SQRT2		= 1.41421356237309504880168872420969808;	// The square root of two (sqrt(2))
	_CONSTANT float M_SQRT1_2	= 0.70710678118654752440084436210484903;	// The reciprocal of the square root of two (1/sqrt(2))

    template<Numeric_Type(T)>
    GLM_FUNC_DECL GLM_CONSTEXPR const T& min(const T& a, const T& b)
    {
        return (b < a) ? b : a;
    }

    template<Numeric_Type(T)>
    GLM_FUNC_DECL GLM_CONSTEXPR const T& max(const T& a, const T& b)
    {
        return (b > a) ? b : a;
    }

    template<Numeric_Type(T)> 
    CLM_FUNC_DECL CLM_CONSTEXPR T remap(const T h1, const T l1, const T h2, const T l2, const T v)
    {
        return l2 + (v - l1) * (h2 - l2) / (h1 - l1);
    }

    template<Numeric_Type(T)> 
    CLM_FUNC_DECL CLM_CONSTEXPR T frac(const T v)
    {
        return v - ::floor(v);
    }

    template<Numeric_Type(T)> 
    CLM_FUNC_DECL CLM_CONSTEXPR T clamp(const T f, const T a, const T b)
    {
        return max(a, min(f, b));
    }

    template <Numeric_Type(T)> 
    CLM_FUNC_DECL CLM_CONSTEXPR T mix(const T a, const T b, const T t)
    {
        return a * (static_cast<T>(1) - t) + b * t;
    }

    template<Numeric_Type(T)> 
    CLM_FUNC_DECL CLM_CONSTEXPR T smooth_step(const T a, const T b, const T x)
    {
        T y = clamp((x - a) / (b - a), 0, 1);
        return (y * y * (static_cast<T>(3) - (static_cast<T>(2) * y)));
    }
}
#endif // _CML_MATH_