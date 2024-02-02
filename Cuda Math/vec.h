#ifndef _CML_VEC_
#define _CML_VEC_

#include "cuda_common.h"
#include "gl_common.h"

#include "numeric.h"

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

#include "ASA.hpp"

namespace cml
{
	template <ASX::ID, typename T> struct vec4_t;
	template <ASX::ID, typename T> struct vec3_t;
	template <ASX::ID, typename T> struct vec2_t;

	template <typename T> using vec4 = vec4_t<ASX::ID_value, T>;
	template <typename T> using vec3 = vec3_t<ASX::ID_value, T>;
	template <typename T> using vec2 = vec2_t<ASX::ID_value, T>;

	typedef vec4<double>	vec4d;
	typedef vec4<float>		vec4f;
	typedef vec4<int32_t>	vec4i;
	typedef vec4<uint32_t>	vec4u;

	typedef vec3<double>	vec3d;
	typedef vec3<float>		vec3f;
	typedef vec3<int32_t>	vec3i;
	typedef vec3<uint32_t>	vec3u;

	typedef vec2<double>	vec2d;
	typedef vec2<float>		vec2f;
	typedef vec2<int32_t>	vec2i;
	typedef vec2<uint32_t>	vec2u;

	template <ASX::ID t_id, typename T>
	struct CML_ALIGN(sizeof(T) * 4) vec4_t
	{
		static_assert(std::is_arithmetic<T>::value, "vec4 type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T x, r, s; ASX_ASA dummy1; };
		union { T y, g, t; ASX_ASA dummy2; };
		union { T z, b, p; ASX_ASA dummy3; };
		union { T w, a, q; ASX_ASA dummy4; };

		/* ------------------------------ Constructors ------------------------------ */

		CML_FUNC_DECL CML_CONSTEXPR vec4_t() : x(0), y(0), z(0), w(0) {};

		CML_FUNC_DECL CML_CONSTEXPR vec4_t(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {};

		CML_FUNC_DECL CML_CONSTEXPR vec4_t(T s) : x(s), y(s), z(s), w(s) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const vec4_t<u_id, U>&v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const vec3_t<u_id, U>&v, T w) : x(v.x), y(v.y), z(v.z), w(w) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const vec3_t<u_id, U>&v) : x(v.x), y(v.y), z(v.z), w(0) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const vec2_t<u_id, U>&v, T z, T w) : x(v.x), y(v.y), z(z), w(w) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const vec2_t<u_id, U>&v) : x(v.x), y(v.y), z(0), w(0) {};


		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const float4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const float3& v) : x(v.x), y(v.y), z(v.z), w(0) {};

		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const float2& v) : x(v.x), y(v.y), z(0), w(0) {};


		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const glm::vec4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const glm::vec3& v) : x(v.x), y(v.y), z(v.z), w(0) {};

		CML_FUNC_DECL CML_CONSTEXPR vec4_t(const glm::vec2& v) : x(v.x), y(v.y), z(0), w(0) {};

		/* ------------------------------- Assignment ------------------------------- */

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator=(const vec4_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			w = static_cast<T>(other.w);
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator=(const vec3_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			w = 0;
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator=(const vec2_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = 0;
			w = 0;
			return *this;
		};

		/* --------------------------------- Casting -------------------------------- */

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec4_t<u_id, U>() const
		{
			return vec4<T>(x, y, z, w);
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec3_t<u_id, U>() const
		{
			return vec3<T>(x, y, z);
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec2_t<u_id, U>() const
		{
			return vec2<T>(x, y);
		};


		CML_FUNC_DECL CML_CONSTEXPR operator float4() const
		{
			return make_float4(x, y, z, w);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator float3() const
		{
			return make_float3(x, y, z);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator float2() const
		{
			return make_float2(x, y);
		};


		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec4() const
		{
			return glm::vec4(x, y, z, w);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec3() const
		{
			return glm::vec3(x, y, z);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};

		/* ---------------------------------- Util ---------------------------------- */

		CML_FUNC_DECL CML_CONSTEXPR void print() const
		{
			printf("(%f, %f, %f, %f)\n", (float)x, (float)y, (float)z, (float)w);
		};
	};

	template <ASX::ID t_id, typename T>
	struct CML_ALIGN(sizeof(T) * 4) vec3_t
	{
		static_assert(std::is_arithmetic<T>::value, "vec3 type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T x, r, s; ASX_ASA dummy1; };
		union { T y, g, t; ASX_ASA dummy2; };
		union { T z, b, p; ASX_ASA dummy3; };

		CML_FUNC_DECL CML_CONSTEXPR vec3_t() : x(0), y(0), z(0) {};

		CML_FUNC_DECL CML_CONSTEXPR vec3_t(T x, T y, T z) : x(x), y(y), z(z) {};

		CML_FUNC_DECL CML_CONSTEXPR vec3_t(T s) : x(s), y(s), z(s) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const vec4_t<u_id, U>&v) : x(v.x), y(v.y), z(v.z) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const vec3_t<u_id, U>&v) : x(v.x), y(v.y), z(v.z) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const vec2_t<u_id, U>&v, T z) : x(v.x), y(v.y), z(z) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const vec2_t<u_id, U>&v) : x(v.x), y(v.y), z(0) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator=(const vec4_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator=(const vec3_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator=(const vec2_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = 0;
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec4_t<u_id, U>() const
		{
			return vec4<T>(x, y, z, 0);
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec3_t<u_id, U>() const
		{
			return vec3<T>(x, y, z);
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec2_t<u_id, U>() const
		{
			return vec2<T>(x, y);
		};


		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const float4 & v) : x(v.x), y(v.y), z(v.z) {};

		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const float3 & v) : x(v.x), y(v.y), z(v.z) {};

		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const float2 & v) : x(v.x), y(v.y), z(0) {};


		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const glm::vec4 & v) : x(v.x), y(v.y), z(v.z) {};

		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const glm::vec3 & v) : x(v.x), y(v.y), z(v.z) {};

		CML_FUNC_DECL CML_CONSTEXPR vec3_t(const glm::vec2 & v) : x(v.x), y(v.y), z(0) {};


		CML_FUNC_DECL CML_CONSTEXPR operator float4() const
		{
			return make_float4(x, y, z, 0);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator float3() const
		{
			return make_float3(x, y, z);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator float2() const
		{
			return make_float2(x, y);
		};


		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec4() const
		{
			return glm::vec4(x, y, z, 0);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec3() const
		{
			return glm::vec3(x, y, z);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};


		CML_FUNC_DECL CML_CONSTEXPR void print() const
		{
			printf("(%f, %f, %f)\n", (float)x, (float)y, (float)z);
		};
	};

	template <ASX::ID t_id, typename T>
	struct CML_ALIGN(sizeof(T) * 2) vec2_t
	{
		static_assert(std::is_arithmetic<T>::value, "vec2 type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T x, r, s; ASX_ASA dummy1; };
		union { T y, g, t; ASX_ASA dummy2; };

		CML_FUNC_DECL CML_CONSTEXPR vec2_t() : x(0), y(0) {};

		CML_FUNC_DECL CML_CONSTEXPR vec2_t(T x, T y) : x(x), y(y) {};

		CML_FUNC_DECL CML_CONSTEXPR vec2_t(T s) : x(s), y(s) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const vec4_t<u_id, U>&v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const vec3_t<u_id, U>&v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const vec2_t<u_id, U>&v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator=(const vec4_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator=(const vec3_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator=(const vec2_t<u_id, U>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec4_t<u_id, U>() const
		{
			return vec4<T>(x, y, 0, 0);
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec3_t<u_id, U>() const
		{
			return vec3<T>(x, y, 0);
		};

		template<ASX::ID u_id, class U>
		CML_FUNC_DECL CML_CONSTEXPR operator vec2_t<u_id, U>() const
		{
			return vec2<T>(x, y);
		};


		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const float4 & v) : x(v.x), y(v.y) {};

		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const float3 & v) : x(v.x), y(v.y) {};

		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const float2 & v) : x(v.x), y(v.y) {};


		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const glm::vec4 & v) : x(v.x), y(v.y) {};

		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const glm::vec3 & v) : x(v.x), y(v.y) {};

		CML_FUNC_DECL CML_CONSTEXPR vec2_t(const glm::vec2 & v) : x(v.x), y(v.y) {};


		CML_FUNC_DECL CML_CONSTEXPR operator float4() const
					{
			return make_float4(x, y, 0, 0);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator float3() const
		{
			return make_float3(x, y, 0);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator float2() const
		{
			return make_float2(x, y);
		};


		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec4() const
		{
			return glm::vec4(x, y, 0, 0);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec3() const
		{
			return glm::vec3(x, y, 0);
		};

		CML_FUNC_DECL CML_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};

		CML_FUNC_DECL CML_CONSTEXPR void print() const
		{
			printf("(%f, %f)\n", (float)x, (float)y);
		};
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator<(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator<=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator>(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator>=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator<(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator<=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator>(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator>=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator==(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator<(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator<=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator>(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR bool operator>=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator*=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator*=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator*(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator*(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator*(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator*(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator*(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator*(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator*(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator*(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator*(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator/(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator/(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator/(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator/(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator/(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator/(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator/(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator/(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator/(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator+(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator+(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator+(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator+(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator+(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator+(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator+(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator+(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator+(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator-(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator-(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator-(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator-(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator-(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator-(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator-(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator-(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator-(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                 Dot Product                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR T dot(const vec4_t<t_id, T>& a, const vec4_t<u_id, T>& b);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR T dot(const vec3_t<t_id, T>& a, const vec3_t<u_id, T>& b);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR T dot(const vec2_t<t_id, T>& a, const vec2_t<u_id, T>& b);

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator-(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator-(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator-(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> floor(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> floor(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> floor(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> ceil(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> ceil(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> ceil(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> frac(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> frac(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> frac(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                     Abs                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> abs(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> abs(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> abs(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> clamp(const vec4_t<t_id, T>& v, const vec4_t<u_id, T>& min, const vec4_t<v_id, T>& max);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> clamp(const vec4_t<t_id, T>& v, const T min, const T max);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> clamp(const vec3_t<t_id, T>& v, const vec3_t<u_id, T>& min, const vec3_t<v_id, T>& max);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> clamp(const vec3_t<t_id, T>& v, const T min, const T max);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> clamp(const vec2_t<t_id, T>& v, const vec2_t<u_id, T>& min, const vec2_t<v_id, T>& max);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> clamp(const vec2_t<t_id, T>& v, const T min, const T max);

	/* -------------------------------------------------------------------------- */
	/*                                     Max                                    */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> max(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> max(const T x, const vec4_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> max(const vec4_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> max(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> max(const T x, const vec3_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> max(const vec3_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> max(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> max(const T x, const vec2_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> max(const vec2_t<t_id, T>& x, const T y);


	/* -------------------------------------------------------------------------- */
	/*                                     Min                                    */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> min(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> min(const T x, const vec4_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> min(const vec4_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> min(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> min(const T x, const vec3_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> min(const vec3_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> min(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> min(const T x, const vec2_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> min(const vec2_t<t_id, T>& x, const T y);


	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec4<T> pow(const vec4_t<t_id, T>& v, const T p);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> pow(const vec3_t<t_id, T>& v, const T p);

	template <ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec2<T> pow(const vec2_t<t_id, T>& v, const T p);


	/* -------------------------------------------------------------------------- */
	/*                         Length, Distance, Normalize                        */
	/* -------------------------------------------------------------------------- */

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR T length(const vec_t<t_id, T>& v);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec_t<ASX::ID_value, T> normalize(const vec_t<t_id, T>& v);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR T distance(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b);

	template <
		template<ASX::ID, class> class vec_t,
		ASX::ID t_id,
		ASX::ID u_id,
		ASX::ID v_id,
		ASX::ID w_id,
		ASX::ID x_id,
		class T>
	CML_FUNC_DECL CML_CONSTEXPR auto remap(const vec_t<t_id, T>& h1, const vec_t<u_id, T>& l1, const vec_t<v_id, T>& h2, const vec_t<w_id, T>& l2, const vec_t<x_id, T>& v);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR auto mix(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b, const float t);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR auto smooth_step(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b, const float x);

	/* -------------------------------------------------------------------------- */
	/*                               Cross, Reflect                               */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> reflect(const vec3_t<t_id, T>& i, const vec3_t<u_id, T>& n);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CML_FUNC_DECL CML_CONSTEXPR vec3<T> cross(const vec3_t<t_id, T>& a, const vec3_t<u_id, T>& b);

	template<class T>
	CML_FUNC_DECL CML_CONSTEXPR T luminance(const T r, const T g, const T b);

	static CML_FUNC_DECL float luminance(const vec3f& color);

	template <ASX::ID t_id, class T>
	static CML_FUNC_DECL vec3<T> gram_schmidt(const vec3_t<t_id, T>& v);

	static CML_FUNC_DECL vec2f sample_spherical_map(const vec3f& d);

	static CML_FUNC_DECL vec3f sample_spherical_direction(const vec2f& uv);
} // namespace cml

#include "vec.inl"

#endif // _CML_VEC_