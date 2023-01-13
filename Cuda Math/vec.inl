#include "vec.h"

namespace cml
{
	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator==(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator!=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z || v1.w != v2.w);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y&& v1.z < v2.z&& v1.w < v2.w);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z && v1.w <= v2.w);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z && v1.w > v2.w);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z && v1.w >= v2.w);
	};

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator==(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator!=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y&& v1.z < v2.z);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z);
	};

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator==(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator!=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y);
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y);
	};

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator*=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		v1.x *= v2.x; v1.y *= v2.y; v1.z *= v2.z; v1.w *= v2.w;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator*=(vec4_t<t_id, T>& v, const T s)
	{
		v.x *= s; v.y *= s; v.z *= s; v.w *= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator*(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		vec4_t<ASX::ID_value, T> tmp(v1);
		return tmp *= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator*(const T s, const vec4_t<t_id, T>& v)
	{
		vec4_t<ASX::ID_value, T> tmp(v);
		return tmp *= s;
	};

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator*(const vec4_t<t_id, T>& v, const T s)
	{
		vec4_t<ASX::ID_value, T> tmp(v);
		return tmp *= s;
	};

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		v1.x *= v2.x; v1.y *= v2.y; v1.z *= v2.z;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v, const T s)
	{
		v.x *= s; v.y *= s; v.z *= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator*(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		vec3_t<ASX::ID_value, T> tmp(v1);
		return tmp *= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator*(const T s, const vec3_t<t_id, T>& v)
	{
		vec3_t<ASX::ID_value, T> tmp(v);
		return tmp *= s;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator*(const vec3_t<t_id, T>& v, const T s)
	{
		vec3_t<ASX::ID_value, T> tmp(v);
		return tmp *= s;
	};

/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		v1.x *= v2.x; v1.y *= v2.y;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v, const T s)
	{
		v.x *= s; v.y *= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator*(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		vec2_t<ASX::ID_value, T> tmp(v1);
		return tmp *= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator*(const T s, const vec2_t<t_id, T>& v)
	{
		vec2_t<ASX::ID_value, T> tmp(v);
		return tmp *= s;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator*(const vec2_t<t_id, T>& v, const T s)
	{
		vec2_t<ASX::ID_value, T> tmp(v);
		return tmp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		v1.x /= v2.x; v1.y /= v2.y; v1.z /= v2.z; v1.w /= v2.w;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v, const T s)
	{
		v.x /= s; v.y /= s; v.z /= s; v.w /= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator/(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		vec4_t<ASX::ID_value, T> tmp(v1);
		return tmp /= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator/(const T s, const vec4_t<t_id, T>& v)
	{
		vec4_t<ASX::ID_value, T> tmp_v(v);
		vec4_t<ASX::ID_value, T> tmp_s(s);
		return tmp_s /= tmp_v;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator/(const vec4_t<t_id, T>& v, const T s)
	{
		vec4_t<ASX::ID_value, T> tmp(v);
		return tmp /= s;
	};

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		v1.x /= v2.x; v1.y /= v2.y; v1.z /= v2.z;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v, const T s)
	{
		v.x /= s; v.y /= s; v.z /= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator/(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		vec3_t<ASX::ID_value, T> tmp(v1);
		return tmp /= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator/(const T s, const vec3_t<t_id, T>& v)
	{
		vec3_t<ASX::ID_value, T> tmp_v(v);
		vec3_t<ASX::ID_value, T> tmp_s(s);
		return tmp_s /= tmp_v;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator/(const vec3_t<t_id, T>& v, const T s)
	{
		vec3_t<ASX::ID_value, T> tmp(v);
		return tmp /= s;
	};

	/* ---------------------------------- vec2 ---------------------------------- */	

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		v1.x /= v2.x; v1.y /= v2.y;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v, const T s)
	{
		v.x /= s; v.y /= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator/(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		vec2_t<ASX::ID_value, T> tmp(v1);
		return tmp /= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator/(const T s, const vec2_t<t_id, T>& v)
	{
		vec2_t<ASX::ID_value, T> tmp_v(v);
		vec2_t<ASX::ID_value, T> tmp_s(s);
		return tmp_s /= tmp_v;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator/(const vec2_t<t_id, T>& v, const T s)
	{
		vec2_t<ASX::ID_value, T> tmp(v);
		return tmp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; v1.w += v2.w;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v, const T s)
	{
		v.x += s; v.y += s; v.z += s; v.w += s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator+(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		vec4_t<ASX::ID_value, T> tmp(v1);
		return tmp += v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator+(const T s, const vec4_t<t_id, T>& v)
	{
		vec4_t<ASX::ID_value, T> tmp(v);
		return tmp += s;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator+(const vec4_t<t_id, T>& v, const T s)
	{
		vec4_t<ASX::ID_value, T> tmp(v);
		return tmp += s;
	};
	
	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		v1.x += v2.x; v1.y += v2.y; v1.z += v2.z;
		return v1;
	};
		
	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v, const T s)
	{
		v.x += s; v.y += s; v.z += s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator+(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		vec3_t<ASX::ID_value, T> tmp(v1);
		return tmp += v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator+(const T s, const vec3_t<t_id, T>& v)
	{
		vec3_t<ASX::ID_value, T> tmp(v);
		return tmp += s;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator+(const vec3_t<t_id, T>& v, const T s)
	{
		vec3_t<ASX::ID_value, T> tmp(v);
		return tmp += s;
	};

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		v1.x += v2.x; v1.y += v2.y;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v, const T s)
	{
		v.x += s; v.y += s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator+(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		vec2_t<ASX::ID_value, T> tmp(v1);
		return tmp += v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator+(const T s, const vec2_t<t_id, T>& v)
	{
		vec2_t<ASX::ID_value, T> tmp(v);
		return tmp += s;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator+(const vec2_t<t_id, T>& v, const T s)
	{
		vec2_t<ASX::ID_value, T> tmp(v);
		return tmp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; v1.w -= v2.w;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v, const T s)
	{
		v.x -= s; v.y -= s; v.z -= s; v.w -= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2)
	{
		vec4_t<ASX::ID_value, T> tmp(v1);
		return tmp -= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const T s, const vec4_t<t_id, T>& v)
	{
		vec4_t<ASX::ID_value, T> tmp_v(v);
		vec4_t<ASX::ID_value, T> tmp_s(s);
		return tmp_s -= tmp_v;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const vec4_t<t_id, T>& v, const T s)
	{
		vec4_t<ASX::ID_value, T> tmp(v);
		return tmp -= s;
	};

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v, const T s)
	{
		v.x -= s; v.y -= s; v.z -= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2)
	{
		vec3_t<ASX::ID_value, T> tmp(v1);
		return tmp -= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const T s, const vec3_t<t_id, T>& v)
	{
		vec3_t<ASX::ID_value, T> tmp_v(v);
		vec3_t<ASX::ID_value, T> tmp_s(s);
		return tmp_s -= tmp_v;
	};

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const vec3_t<t_id, T>& v, const T s)
	{
		vec3_t<ASX::ID_value, T> tmp(v);
		return tmp -= s;
	};

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		v1.x -= v2.x; v1.y -= v2.y;
		return v1;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v, const T s)
	{
		v.x -= s; v.y -= s;
		return v;
	};

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2)
	{
		vec2_t<ASX::ID_value, T> tmp(v1);
		return tmp -= v2;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const T s, const vec2_t<t_id, T>& v)
	{
		vec2_t<ASX::ID_value, T> tmp_v(v);
		vec2_t<ASX::ID_value, T> tmp_s(s);
		return tmp_s -= tmp_v;
	};

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const vec2_t<t_id, T>& v, const T s)
	{
		vec2_t<ASX::ID_value, T> tmp(v);
		return tmp -= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Dot Product                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR T dot(const vec4_t<t_id, T>& a, const vec4_t<u_id, T>& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T dot(const vec3_t<t_id, T>& a, const vec3_t<u_id, T>& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T dot(const vec2_t<t_id, T>& a, const vec2_t<u_id, T>& b)
	{
		return (a.x * b.x) + (a.y * b.y);
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(-v.x, -v.y, -v.z, -v.w);
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(-v.x, -v.y, -v.z);
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(-v.x, -v.y);
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> floor(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(::floor(v.x), ::floor(v.y), ::floor(v.z), ::floor(v.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> floor(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(::floor(v.x), ::floor(v.y), ::floor(v.z));
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> floor(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(::floor(v.x), ::floor(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> ceil(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(::ceil(v.x), ::ceil(v.y), ::ceil(v.z), ::ceil(v.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> ceil(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(::ceil(v.x), ::ceil(v.y), ::ceil(v.z));
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> ceil(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(::ceil(v.x), ::ceil(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> frac(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(frac(v.x), frac(v.y), frac(v.z), frac(v.w));
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> frac(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(frac(v.x), frac(v.y), frac(v.z));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> frac(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(frac(v.x), frac(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Abs                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> abs(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(::abs(v.x), ::abs(v.y), ::abs(v.z), ::abs(v.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> abs(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(::abs(v.x), ::abs(v.y), ::abs(v.z));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> abs(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(::abs(v.x), ::abs(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> clamp(const vec4_t<t_id, T>& v, const vec4_t<u_id, T>& min, const vec4_t<v_id, T>& max)
	{
		return vec4_t<ASX::ID_value, T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y), clamp(v.z, min.z, max.z), clamp(v.w, min.w, max.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> clamp(const vec4_t<t_id, T>& v, const T min, const T max)
	{
		return vec4_t<ASX::ID_value, T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max), clamp(v.w, min, max));
	}

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> clamp(const vec3_t<t_id, T>& v, const vec3_t<u_id, T>& min, const vec3_t<v_id, T>& max)
	{
		return vec3_t<ASX::ID_value, T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y), clamp(v.z, min.z, max.z));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> clamp(const vec3_t<t_id, T>& v, const T min, const T max)
	{
		return vec3_t<ASX::ID_value, T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max));
	}

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> clamp(const vec2_t<t_id, T>& v, const vec2_t<u_id, T>& min, const vec2_t<v_id, T>& max)
	{
		return vec2_t<ASX::ID_value, T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> clamp(const vec2_t<t_id, T>& v, const T min, const T max)
	{
		return vec2_t<ASX::ID_value, T>(clamp(v.x, min, max), clamp(v.y, min, max));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Max                                    */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> max(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z), max(x.w, y.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> max(const T x, const vec4_t<t_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(max(x, y.x), max(x, y.y), max(x, y.z), max(x, y.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> max(const vec4_t<t_id, T>& x, const T y)
	{
		return vec4_t<ASX::ID_value, T>(max(x.x, y), max(x.y, y), max(x.z, y), max(x.w, y));
	}

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> max(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(max(x.x, y.x), max(x.y, y.y), max(x.z, y.z));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> max(const T x, const vec3_t<t_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(max(x, y.x), max(x, y.y), max(x, y.z));
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> max(const vec3_t<t_id, T>& x, const T y)
	{
		return vec3_t<ASX::ID_value, T>(max(x.x, y), max(x.y, y), max(x.z, y));
	}

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> max(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(max(x.x, y.x), max(x.y, y.y));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> max(const T x, const vec2_t<t_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(max(x, y.x), max(x, y.y));
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> max(const vec2_t<t_id, T>& x, const T y)
	{
		return vec2_t<ASX::ID_value, T>(max(x.x, y), max(x.y, y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Min                                    */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> min(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(min(x.x, y.x), min(x.y, y.y), min(x.z, y.z), min(x.w, y.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> min(const T x, const vec4_t<t_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(min(x, y.x), min(x, y.y), min(x, y.z), min(x, y.w));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> min(const vec4_t<t_id, T>& x, const T y)
	{
		return vec4_t<ASX::ID_value, T>(min(x.x, y), min(x.y, y), min(x.z, y), min(x.w, y));
	}

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> min(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(min(x.x, y.x), min(x.y, y.y), min(x.z, y.z));
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> min(const T x, const vec3_t<t_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(min(x, y.x), min(x, y.y), min(x, y.z));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> min(const vec3_t<t_id, T>& x, const T y)
	{
		return vec3_t<ASX::ID_value, T>(min(x.x, y), min(x.y, y), min(x.z, y));
	}

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> min(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(min(x.x, y.x), min(x.y, y.y));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> min(const T x, const vec2_t<t_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(min(x, y.x), min(x, y.y));
	}

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> min(const vec2_t<t_id, T>& x, const T y)
	{
		return vec2_t<ASX::ID_value, T>(min(x.x, y), min(x.y, y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> pow(const vec4_t<t_id, T>& v, const T p)
	{
		return vec4_t<ASX::ID_value, T>(::pow(v.x, p), ::pow(v.y, p), ::pow(v.z, p), ::pow(v.w, p));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> pow(const vec3_t<t_id, T>& v, const T p)
	{
		return vec3_t<ASX::ID_value, T>(::pow(v.x, p), ::pow(v.y, p), ::pow(v.z, p));
	}

	template <ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> pow(const vec2_t<t_id, T>& v, const T p)
	{
		return vec2_t<ASX::ID_value, T>(::pow(v.x, p), ::pow(v.y, p));
	}

	/* -------------------------------------------------------------------------- */
	/*                         Length, Distance, Normalize                        */
	/* -------------------------------------------------------------------------- */

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR T length(const vec_t<t_id, T>& v)
	{
		return ::sqrt(dot(v, v));
	}

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec_t<ASX::ID_value, T> normalize(const vec_t<t_id, T>& v)
	{
		vec_t<ASX::ID_value, T> tmp(v);
		T l = length(v);
		return (l == static_cast<T>(0)) ? tmp : tmp / l;
	}

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T distance(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b)
	{
		return length(b - a);
	}

	template <
		template<ASX::ID, class> class vec_t,
		ASX::ID t_id,
		ASX::ID u_id,
		ASX::ID v_id,
		ASX::ID w_id,
		ASX::ID x_id,
		class T>
	CLM_FUNC_DECL CLM_CONSTEXPR auto remap(const vec_t<t_id, T>& h1, const vec_t<u_id, T>& l1, const vec_t<v_id, T>& h2, const vec_t<w_id, T>& l2, const vec_t<x_id, T>& v)
	{
		return l2 + (v - l1) * (h2 - l2) / (h1 - l1);
	}

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR auto mix(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b, const T t)
	{
		return a * (static_cast<T>(1) - t) + b * t;
	}

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR auto smooth_step(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b, const T x)
	{
		float y = clamp((x - a) / (b - a), static_cast<T>(0), static_cast<T>(1));
		return (y * y * (static_cast<T>(3) - (static_cast<T>(2) * y)));
	}

	/* -------------------------------------------------------------------------- */
	/*                               Cross, Reflect                               */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> reflect(const vec3_t<t_id, T>& i, const vec3_t<u_id, T>& n)
	{
		return i - static_cast<T>(2) * n * dot(n, i);
	}

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> cross(const vec3_t<t_id, T>& a, const vec3_t<u_id, T>& b)
	{
		return vec3_t<ASX::ID_value, T>(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
	}

	template<class T> 
	CLM_FUNC_DECL CLM_CONSTEXPR T luminance(const T r, const T g, const T b)
	{
		return (static_cast<T>(0.299) * r + static_cast<T>(0.587) * g + static_cast<T>(0.114) * b);
	}

	template<class T>
	static CLM_FUNC_DECL float luminance(const vec3<T>& color)
	{
		return (static_cast<T>(0.299) * color.x + static_cast<T>(0.587) * color.y + static_cast<T>(0.114) * color.z);
	}

	template <class T>
	static CLM_FUNC_DECL vec3<T> gram_schmidt(const vec3<T>& v)
	{
		vec3<T> x(rand_float(-1, 1), rand_float(-1, 1), rand_float(-1, 1));

		float x_dot_v = dot(x, v);

		vec3<T> v_norm = normalize(v);
		vec3<T> v_norm_2 = v_norm * v_norm;

		x = x - ((x_dot_v * v) / v_norm_2);

		return normalize(x);
	}

	static CLM_FUNC_DECL vec2f sample_spherical_map(const vec3f& d)
	{
		vec2f uv = vec2f(0.5f + ::atan2(d.z, d.x) * M_1_2PI, 0.5f - ::asin(d.y) * M_1_PI);
		return uv;
	}

	static CLM_FUNC_DECL vec3f sample_spherical_direction(const vec2f& uv)
	{
		float phi = 2.f * M_PI * (uv.x - 0.5f);
		float theta = M_PI * uv.y;

		vec3f n;
		n.x = ::cos(phi) * ::sin(theta);
		n.z = ::sin(phi) * ::sin(theta);
		n.y = ::cos(theta);

		return n;
	}
} // namespace cml