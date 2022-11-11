#ifndef _JEK_MATH_
#define _JEK_MATH_

#include "CudaCommon.h"

namespace jek
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
	_CONSTANT float M_1_PI		= 0.31830988618379067153776752674502872;	// The reciprocal of pi (1/pi)
	_CONSTANT float M_1_2PI		= 0.15915494309189533576888376337251436;	// The reciprocal of 2pi (1/2pi)
	_CONSTANT float M_1_4PI		= 0.07957747154594766788444188168625718;	// The reciprocal of 4pi (1/4pi)
	_CONSTANT float M_2_PI		= 0.63661977236758134307553505349005744;	// Two times the reciprocal of pi (2/pi)
	_CONSTANT float M_2_SQRTPI	= 1.12837916709551257389615890312154517;	// Two times the reciprocal of the square root of pi (2/sqrt(pi))
	_CONSTANT float M_SQRT2		= 1.41421356237309504880168872420969808;	// The square root of two (sqrt(2))
	_CONSTANT float M_SQRT1_2	= 0.70710678118654752440084436210484903;	// The reciprocal of the square root of two (1/sqrt(2))

    template<class T> _HOST_DEVICE 
    inline T remap(const T h1, const T l1, const T h2, const T l2, const T v)
    {
        return l2 + (v - l1) * (h2 - l2) / (h1 - l1);
    }

    template<class T> _HOST_DEVICE
    inline T frac(const T v)
    {
        return v - ::floor(v);
    }

    template<class T> _HOST_DEVICE
    inline T clamp(const T f, const T a, const T b)
    {
        return ::fmax(a, ::fmin(f, b));
    }

    template <class T> _HOST_DEVICE
    inline T mix(const T a, const T b, const T t)
    {
        return a * (1.f - t) + b * t;
    }

    template<class T> _HOST_DEVICE 
    T smooth_step(const T a, const T b, const T x)
    {
        T y = clamp((x - a) / (b - a), 0, 1);
        return (y * y * (T(3) - (T(2) * y)));
    }
}
#endif // _JEK_MATH_