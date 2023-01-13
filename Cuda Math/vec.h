#pragma once

#include "CudaCommon.h"
#include "GLCommon.h"
#include "Random.h"
#include "Numeric.h"

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

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t() : x(0), y(0), z(0), w(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(T s) : x(s), y(s), z(s), w(s) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const vec4_t<u_id, T>&v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const vec3_t<u_id, T>&v, T w) : x(v.x), y(v.y), z(v.z), w(w) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const vec3_t<u_id, T>&v) : x(v.x), y(v.y), z(v.z), w(0) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const vec2_t<u_id, T>&v, T z, T w) : x(v.x), y(v.y), z(z), w(w) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const vec2_t<u_id, T>&v) : x(v.x), y(v.y), z(0), w(0) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator=(const vec4_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			w = static_cast<T>(other.w);
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator=(const vec3_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			w = 0;
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator=(const vec2_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = 0;
			w = 0;
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec4_t<u_id, T>() const
		{
			return vec4_t<ASX::ID_value, T>(x, y, z, w);
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec3_t<u_id, T>() const
		{
			return vec3_t<ASX::ID_value, T>(x, y, z);
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec2_t<u_id, T>() const
		{
			return vec2_t<ASX::ID_value, T>(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const float4 & v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const float3 & v) : x(v.x), y(v.y), z(v.z), w(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const float2 & v) : x(v.x), y(v.y), z(0), w(0) {};


		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const glm::vec4 & v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const glm::vec3 & v) : x(v.x), y(v.y), z(v.z), w(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const glm::vec2 & v) : x(v.x), y(v.y), z(0), w(0) {};


		CLM_FUNC_DECL CLM_CONSTEXPR operator float4() const
		{
			return make_float4(x, y, z, w);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator float3() const
		{
			return make_float3(x, y, z);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator float2() const
		{
			return make_float2(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec4() const
		{
			return glm::vec4(x, y, z, w);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec3() const
		{
			return glm::vec3(x, y, z);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR void print() const
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

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t() : x(0), y(0), z(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(T x, T y, T z) : x(x), y(y), z(z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(T s) : x(s), y(s), z(s) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec4_t<u_id, T>&v) : x(v.x), y(v.y), z(v.z) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec3_t<u_id, T>&v) : x(v.x), y(v.y), z(v.z) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec2_t<u_id, T>&v, T z) : x(v.x), y(v.y), z(z) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec2_t<u_id, T>&v) : x(v.x), y(v.y), z(0) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator=(const vec4_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator=(const vec3_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = static_cast<T>(other.z);
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator=(const vec2_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			z = 0;
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec4_t<u_id, T>() const
		{
			return vec4_t<ASX::ID_value, T>(x, y, z, 0);
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec3_t<u_id, T>() const
		{
			return vec3_t<ASX::ID_value, T>(x, y, z);
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec2_t<u_id, T>() const
		{
			return vec2_t<ASX::ID_value, T>(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const float4 & v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const float3 & v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const float2 & v) : x(v.x), y(v.y), z(0) {};


		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const glm::vec4 & v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const glm::vec3 & v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const glm::vec2 & v) : x(v.x), y(v.y), z(0) {};


		CLM_FUNC_DECL CLM_CONSTEXPR operator float4() const
		{
			return make_float4(x, y, z, 0);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator float3() const
		{
			return make_float3(x, y, z);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator float2() const
		{
			return make_float2(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec4() const
		{
			return glm::vec4(x, y, z, 0);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec3() const
		{
			return glm::vec3(x, y, z);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR void print() const
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

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t() : x(0), y(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(T x, T y) : x(x), y(y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(T s) : x(s), y(s) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const vec4_t<u_id, T>&v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const vec3_t<u_id, T>&v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const vec2_t<u_id, T>&v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator=(const vec4_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator=(const vec3_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator=(const vec2_t<u_id, T>&other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec4_t<u_id, T>() const
		{
			return vec4_t<ASX::ID_value, T>(x, y, 0, 0);
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec3_t<u_id, T>() const
		{
			return vec3_t<ASX::ID_value, T>(x, y, 0);
		};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec2_t<u_id, T>() const
		{
			return vec2_t<ASX::ID_value, T>(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const float4 & v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const float3 & v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const float2 & v) : x(v.x), y(v.y) {};


		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const glm::vec4 & v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const glm::vec3 & v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const glm::vec2 & v) : x(v.x), y(v.y) {};


		CLM_FUNC_DECL CLM_CONSTEXPR operator float4() const
					{
			return make_float4(x, y, 0, 0);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator float3() const
		{
			return make_float3(x, y, 0);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator float2() const
		{
			return make_float2(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec4() const
		{
			return glm::vec4(x, y, 0, 0);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec3() const
		{
			return glm::vec3(x, y, 0);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};

		CLM_FUNC_DECL CLM_CONSTEXPR void print() const
		{
			printf("(%f, %f)\n", (float)x, (float)y);
		};
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator==(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator!=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator==(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator!=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator==(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator!=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator<=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR bool operator>=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator*=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator*=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator*(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator*(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator*(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator*(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator*(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator*(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator*(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator*(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator*(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator/(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator/(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator/(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator/(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator/(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator/(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator/(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator/(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator/(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator+(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator+(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator+(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator+(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator+(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator+(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator+(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator+(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator+(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const vec4_t<t_id, T>& v1, const vec4_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const T s, const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const vec4_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const vec3_t<t_id, T>& v1, const vec3_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const T s, const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const vec3_t<t_id, T>& v, const T s);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v, const T s);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const vec2_t<t_id, T>& v1, const vec2_t<u_id, T>& v2);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const T s, const vec2_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const vec2_t<t_id, T>& v, const T s);

	/* -------------------------------------------------------------------------- */
	/*                                 Dot Product                                */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T dot(const vec4_t<t_id, T>& a, const vec4_t<u_id, T>& b);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T dot(const vec3_t<t_id, T>& a, const vec3_t<u_id, T>& b);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T dot(const vec2_t<t_id, T>& a, const vec2_t<u_id, T>& b);

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> operator-(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> operator-(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> operator-(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> floor(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> floor(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> floor(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> ceil(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> ceil(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> ceil(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> frac(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> frac(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> frac(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                     Abs                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> abs(const vec4_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> abs(const vec3_t<t_id, T>& v);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> abs(const vec2_t<t_id, T>& v);

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> clamp(const vec4_t<t_id, T>& v, const vec4_t<u_id, T>& min, const vec4_t<v_id, T>& max);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> clamp(const vec4_t<t_id, T>& v, const T min, const T max);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> clamp(const vec3_t<t_id, T>& v, const vec3_t<u_id, T>& min, const vec3_t<v_id, T>& max);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> clamp(const vec3_t<t_id, T>& v, const T min, const T max);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> clamp(const vec2_t<t_id, T>& v, const vec2_t<u_id, T>& min, const vec2_t<v_id, T>& max);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> clamp(const vec2_t<t_id, T>& v, const T min, const T max);

	/* -------------------------------------------------------------------------- */
	/*                                     Max                                    */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> max(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> max(const T x, const vec4_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> max(const vec4_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> max(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> max(const T x, const vec3_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> max(const vec3_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> max(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> max(const T x, const vec2_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> max(const vec2_t<t_id, T>& x, const T y);


	/* -------------------------------------------------------------------------- */
	/*                                     Min                                    */
	/* -------------------------------------------------------------------------- */

	/* ---------------------------------- vec4 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> min(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> min(const T x, const vec4_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> min(const vec4_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec3 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> min(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> min(const T x, const vec3_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> min(const vec3_t<t_id, T>& x, const T y);

	/* ---------------------------------- vec2 ---------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> min(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> min(const T x, const vec2_t<t_id, T>& y);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> min(const vec2_t<t_id, T>& x, const T y);


	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<ASX::ID_value, T> pow(const vec4_t<t_id, T>& v, const T p);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> pow(const vec3_t<t_id, T>& v, const T p);

	template <ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<ASX::ID_value, T> pow(const vec2_t<t_id, T>& v, const T p);


	/* -------------------------------------------------------------------------- */
	/*                         Length, Distance, Normalize                        */
	/* -------------------------------------------------------------------------- */

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T length(const vec_t<t_id, T>& v);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec_t<ASX::ID_value, T> normalize(const vec_t<t_id, T>& v);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T distance(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b);

	template <
		template<ASX::ID, class> class vec_t,
		ASX::ID t_id,
		ASX::ID u_id,
		ASX::ID v_id,
		ASX::ID w_id,
		ASX::ID x_id,
		class T>
	CLM_FUNC_DECL CLM_CONSTEXPR auto remap(const vec_t<t_id, T>& h1, const vec_t<u_id, T>& l1, const vec_t<v_id, T>& h2, const vec_t<w_id, T>& l2, const vec_t<x_id, T>& v);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR auto mix(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b, const float t);

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR auto smooth_step(const vec_t<t_id, T>& a, const vec_t<u_id, T>& b, const float x);

	/* -------------------------------------------------------------------------- */
	/*                               Cross, Reflect                               */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> reflect(const vec3_t<t_id, T>& i, const vec3_t<u_id, T>& n);

	template <ASX::ID t_id, ASX::ID u_id, class T>
	CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<ASX::ID_value, T> cross(const vec3_t<t_id, T>& a, const vec3_t<u_id, T>& b);

	template<class T>
	CLM_FUNC_DECL CLM_CONSTEXPR T luminance(const T r, const T g, const T b);

	static CLM_FUNC_DECL float luminance(const vec3f& color);

	template <ASX::ID t_id, class T>
	static CLM_FUNC_DECL vec3<T> gram_schmidt(const vec3_t<t_id, T>& v);

	static CLM_FUNC_DECL vec2f sample_spherical_map(const vec3f& d);

	static CLM_FUNC_DECL vec3f sample_spherical_direction(const vec2f& uv);
} // namespace cml

#include "vec.inl"