#ifndef _CML_VECTOR_
#define _CML_VECTOR_

#include "CudaCommon.h"
#include "GLCommon.h"
#include "Random.h"
#include "dMath.h"

#include "ASA.hpp"

#define Numeric_Type(T) typename T, typename = typename std::enable_if<std::is_arithmetic<T>::value, T>::type

namespace cml
{
	template <ASX::ID, class T> struct vec4_t;
	template <ASX::ID, class T> struct vec3_t;
	template <ASX::ID, class T> struct vec2_t;

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
	struct _ALIGN(sizeof(T) * 4) vec4_t
	{
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

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

	
		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator=(const vec4_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y); 
			z = static_cast<T>(other.z); 
			w = static_cast<T>(other.w);
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator=(const vec3_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x); 
			y = static_cast<T>(other.y); 
			z = static_cast<T>(other.z); 
			w = 0;
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t<t_id, T>& operator=(const vec2_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x); 
			y = static_cast<T>(other.y); 
			z = 0; 
			w = 0;
			return *this;
		};
		
		
		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec4_t<u_id, U>() const
		{
			return vec4_t<ASX::ID_value, U>(x, y, z, w);
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec3_t<u_id, U>() const
		{
			return vec3_t<ASX::ID_value, U>(x, y, z);
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec2_t<u_id, U>() const
		{
			return vec2_t<ASX::ID_value, U>(x, y);
		};
		

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const float4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const float3& v) : x(v.x), y(v.y), z(v.z), w(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const float2& v) : x(v.x), y(v.y), z(0), w(0) {};


		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const glm::vec4& v) : x(v.x), y(v.y), z(v.z), w(v.w) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const glm::vec3& v) : x(v.x), y(v.y), z(v.z), w(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec4_t(const glm::vec2& v) : x(v.x), y(v.y), z(0), w(0) {};


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
	struct _ALIGN(sizeof(T) * 4) vec3_t
	{
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T x, r, s; ASX_ASA dummy1; };
		union { T y, g, t; ASX_ASA dummy2; };
		union { T z, b, p; ASX_ASA dummy3; };

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t() : x(0), y(0), z(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(T x, T y, T z) : x(x), y(y), z(z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(T s) : x(s), y(s), z(s) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec4_t<u_id, T>& v) : x(v.x), y(v.y), z(v.z) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec3_t<u_id, T>& v) : x(v.x), y(v.y), z(v.z) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec2_t<u_id, T>& v, T z) : x(v.x), y(v.y), z(z) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const vec2_t<u_id, T>& v) : x(v.x), y(v.y), z(0) {};


		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator=(const vec4_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y); 
			z = static_cast<T>(other.z);
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator=(const vec3_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y); 
			z = static_cast<T>(other.z);
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t<t_id, T>& operator=(const vec2_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x); 
			y = static_cast<T>(other.y); 
			z = 0;
			return *this;
		};
		
		
		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec4_t<u_id, U>() const
		{
			return vec4_t<ASX::ID_value, U>(x, y, z, 0);
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec3_t<u_id, U>() const
		{
			return vec3_t<ASX::ID_value, U>(x, y, z);
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec2_t<u_id, U>() const
		{
			return vec2_t<ASX::ID_value, U>(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const float4& v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const float3& v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const float2& v) : x(v.x), y(v.y), z(0) {};


		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const glm::vec4& v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const glm::vec3& v) : x(v.x), y(v.y), z(v.z) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec3_t(const glm::vec2& v) : x(v.x), y(v.y), z(0) {};


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
	struct _ALIGN(sizeof(T) * 2) vec2_t
	{
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		typedef ASX::ASAGroup<T, t_id> ASX_ASA;

		union { T x, r, s; ASX_ASA dummy1; };
		union { T y, g, t; ASX_ASA dummy2; };

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t() : x(0), y(0) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(T x, T y) : x(x), y(y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(T s) : x(s), y(s) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const vec4_t<u_id, T>& v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const vec3_t<u_id, T>& v) : x(v.x), y(v.y) {};

		template<ASX::ID u_id>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const vec2_t<u_id, T>& v) : x(v.x), y(v.y) {};


		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator=(const vec4_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x); 
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator=(const vec3_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t<t_id, T>& operator=(const vec2_t<u_id, U>& other)
		{
			x = static_cast<T>(other.x);
			y = static_cast<T>(other.y);
			return *this;
		};


		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec4_t<u_id, U>() const
		{
			return vec4_t<ASX::ID_value, U>(x, y, 0, 0);
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec3_t<u_id, U>() const
		{
			return vec3_t<ASX::ID_value, U>(x, y, 0);
		};

		template<ASX::ID u_id, Numeric_Type(U)>
		CLM_FUNC_DECL CLM_CONSTEXPR operator vec2_t<u_id, U>() const
		{
			return vec2_t<ASX::ID_value, U>(x, y);
		};


		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const float4& v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const float3& v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const float2& v) : x(v.x), y(v.y) {};


		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const glm::vec4& v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const glm::vec3& v) : x(v.x), y(v.y) {};

		CLM_FUNC_DECL CLM_CONSTEXPR vec2_t(const glm::vec2& v) : x(v.x), y(v.y) {};


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

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator==(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator!=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z || v1.w != v2.w);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator<(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y&& v1.z < v2.z&& v1.w < v2.w);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator<=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z && v1.w <= v2.w);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator>(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z && v1.w > v2.w);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator>=(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z && v1.w >= v2.w);
	};

	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator==(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator!=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator<(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y&& v1.z < v2.z);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator<=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator>(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator>=(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z);
	};

	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator==(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y);
	};
	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator!=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator<(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator<=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator>(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y);
	};
	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	bool operator>=(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y);
	};

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	vec4_t<t_id, T> operator*=(vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		v1.x *= v2.x; v1.y *= v2.y; v1.z *= v2.z; v1.w *= v2.w;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator*(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
		-> vec4_t<ASX::ID_value, decltype(v1.x* v2.x)>
	{
		vec4_t<ASX::ID_value, decltype(v1.x* v2.x)> temp(v1);
		return temp *= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR  
	vec4_t<t_id, T> operator*=(vec4_t<t_id, T>& v, const U s)
	{
		v.x *= s; v.y *= s; v.z *= s; v.w *= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator*(const T s, const vec4_t<t_id, U>& v)
		-> vec4_t<ASX::ID_value, decltype(s * v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s* v.x)> temp(v);
		return temp *= s;
	};
	
	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator*(const vec4_t<t_id, T>& v, const U s)
		-> vec4_t<ASX::ID_value, decltype(s * v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s * v.x)> temp(v);
		return temp *= s;
	};
	

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		v1.x *= v2.x; v1.y *= v2.y; v1.z *= v2.z;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator*(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
		-> vec3_t<ASX::ID_value, decltype(v1.x * v2.x)>
	{
		vec3_t<ASX::ID_value, decltype(v1.x* v2.x)> temp(v1);
		return temp *= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<t_id, T>& operator*=(vec3_t<t_id, T>& v, const U s)
	{
		v.x *= s; v.y *= s; v.z *= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator*(const T s, const vec3_t<t_id, U>& v)
		-> vec3_t<ASX::ID_value, decltype(s * v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s * v.x)> temp(v);
		return temp *= s;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator*(const vec3_t<t_id, T>& v, const U s)
		-> vec3_t<ASX::ID_value, decltype(s * v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s * v.x)> temp(v);
		return temp *= s;
	};


	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		v1.x *= v2.x; v1.y *= v2.y;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator*(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
		-> vec2_t<ASX::ID_value, decltype(v1.x* v2.x)>
	{
		vec2_t<ASX::ID_value, decltype(v1.x* v2.x)> temp(v1);
		return temp *= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator*=(vec2_t<t_id, T>& v, const U s)
	{
		v.x *= s; v.y *= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator*(const T s, const vec2_t<t_id, U>& v)
		-> vec2_t<ASX::ID_value, decltype(s* v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s* v.x)> temp(v);
		return temp *= s;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator*(const vec2_t<t_id, T>& v, const U s)
		-> vec2_t<ASX::ID_value, decltype(s* v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s* v.x)> temp(v);
		return temp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		v1.x /= v2.x; v1.y /= v2.y; v1.z /= v2.z; v1.w /= v2.w;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator/(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
		-> vec4_t<ASX::ID_value, decltype(v1.x/ v2.x)>
	{
		vec4_t<ASX::ID_value, decltype(v1.x/ v2.x)> temp(v1);
		return temp /= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR  
	vec4_t<t_id, T>& operator/=(vec4_t<t_id, T>& v, const U s)
	{
		v.x /= s; v.y /= s; v.z /= s; v.w /= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator/(const T s, const vec4_t<t_id, U>& v)
		-> vec4_t<ASX::ID_value, decltype(s / v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s/ v.x)> temp(v);
		return temp /= s;
	};
	
	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator/(const vec4_t<t_id, T>& v, const U s)
		-> vec4_t<ASX::ID_value, decltype(s / v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s / v.x)> temp(v);
		return temp /= s;
	};


	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		v1.x /= v2.x; v1.y /= v2.y; v1.z /= v2.z;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator/(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
		-> vec3_t<ASX::ID_value, decltype(v1.x/ v2.x)>
	{
		vec3_t<ASX::ID_value, decltype(v1.x/ v2.x)> temp(v1);
		return temp /= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR  
	vec3_t<t_id, T>& operator/=(vec3_t<t_id, T>& v, const U s)
	{
		v.x /= s; v.y /= s; v.z /= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator/(const T s, const vec3_t<t_id, U>& v)
		-> vec3_t<ASX::ID_value, decltype(s / v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s/ v.x)> temp(v);
		return temp /= s;
	};
	
	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator/(const vec3_t<t_id, T>& v, const U s)
		-> vec3_t<ASX::ID_value, decltype(s / v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s / v.x)> temp(v);
		return temp /= s;
	};


	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		v1.x /= v2.x; v1.y /= v2.y;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator/(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
		-> vec2_t<ASX::ID_value, decltype(v1.x / v2.x)>
	{
		vec2_t<ASX::ID_value, decltype(v1.x / v2.x)> temp(v1);
		return temp /= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator/=(vec2_t<t_id, T>& v, const U s)
	{
		v.x /= s; v.y /= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator/(const T s, const vec2_t<t_id, U>& v)
		-> vec2_t<ASX::ID_value, decltype(s / v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s / v.x)> temp(v);
		return temp /= s;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator/(const vec2_t<t_id, T>& v, const U s)
		-> vec2_t<ASX::ID_value, decltype(s / v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s / v.x)> temp(v);
		return temp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		v1.x += v2.x; v1.y += v2.y; v1.z += v2.z; v1.w += v2.w;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator+(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
		-> vec4_t<ASX::ID_value, decltype(v1.x+ v2.x)>
	{
		vec4_t<ASX::ID_value, decltype(v1.x+ v2.x)> temp(v1);
		return temp += v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR  
	vec4_t<t_id, T>& operator+=(vec4_t<t_id, T>& v, const U s)
	{
		v.x += s; v.y += s; v.z += s; v.w += s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator+(const T s, const vec4_t<t_id, U>& v)
		-> vec4_t<ASX::ID_value, decltype(s + v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s+ v.x)> temp(v);
		return temp += s;
	};
	
	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator+(const vec4_t<t_id, T>& v, const U s)
		-> vec4_t<ASX::ID_value, decltype(s + v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s + v.x)> temp(v);
		return temp += s;
	};


	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		v1.x += v2.x; v1.y += v2.y; v1.z += v2.z;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator+(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
		-> vec3_t<ASX::ID_value, decltype(v1.x+ v2.x)>
	{
		vec3_t<ASX::ID_value, decltype(v1.x+ v2.x)> temp(v1);
		return temp += v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR  
	vec3_t<t_id, T>& operator+=(vec3_t<t_id, T>& v, const U s)
	{
		v.x += s; v.y += s; v.z += s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator+(const T s, const vec3_t<t_id, U>& v)
		-> vec3_t<ASX::ID_value, decltype(s + v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s+ v.x)> temp(v);
		return temp += s;
	};
	
	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator+(const vec3_t<t_id, T>& v, const U s)
		-> vec3_t<ASX::ID_value, decltype(s + v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s + v.x)> temp(v);
		return temp += s;
	};


	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		v1.x += v2.x; v1.y += v2.y;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator+(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
		-> vec2_t<ASX::ID_value, decltype(v1.x + v2.x)>
	{
		vec2_t<ASX::ID_value, decltype(v1.x + v2.x)> temp(v1);
		return temp += v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator+=(vec2_t<t_id, T>& v, const U s)
	{
		v.x += s; v.y += s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator+(const T s, const vec2_t<t_id, U>& v)
		-> vec2_t<ASX::ID_value, decltype(s + v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s + v.x)> temp(v);
		return temp += s;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator+(const vec2_t<t_id, T>& v, const U s)
		-> vec2_t<ASX::ID_value, decltype(s + v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s + v.x)> temp(v);
		return temp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
	{
		v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z; v1.w -= v2.w;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator-(const vec4_t<t_id, T>& v1, const vec4_t<u_id, U>& v2)
		-> vec4_t<ASX::ID_value, decltype(v1.x- v2.x)>
	{
		vec4_t<ASX::ID_value, decltype(v1.x- v2.x)> temp(v1);
		return temp -= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR  
	vec4_t<t_id, T>& operator-=(vec4_t<t_id, T>& v, const U s)
	{
		v.x -= s; v.y -= s; v.z -= s; v.w -= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator-(const T s, const vec4_t<t_id, U>& v)
		-> vec4_t<ASX::ID_value, decltype(s - v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s- v.x)> temp(v);
		return temp -= s;
	};
	
	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator-(const vec4_t<t_id, T>& v, const U s)
		-> vec4_t<ASX::ID_value, decltype(s - v.x)>
	{
		vec4_t<ASX::ID_value, decltype(s - v.x)> temp(v);
		return temp -= s;
	};


	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR 
	vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
	{
		v1.x -= v2.x; v1.y -= v2.y; v1.z -= v2.z;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator-(const vec3_t<t_id, T>& v1, const vec3_t<u_id, U>& v2)
		-> vec3_t<ASX::ID_value, decltype(v1.x- v2.x)>
	{
		vec3_t<ASX::ID_value, decltype(v1.x- v2.x)> temp(v1);
		return temp -= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR  
	vec3_t<t_id, T>& operator-=(vec3_t<t_id, T>& v, const U s)
	{
		v.x -= s; v.y -= s; v.z -= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator-(const T s, const vec3_t<t_id, U>& v)
		-> vec3_t<ASX::ID_value, decltype(s - v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s- v.x)> temp(v);
		return temp -= s;
	};
	
	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR 
	auto operator-(const vec3_t<t_id, T>& v, const U s)
		-> vec3_t<ASX::ID_value, decltype(s - v.x)>
	{
		vec3_t<ASX::ID_value, decltype(s - v.x)> temp(v);
		return temp -= s;
	};


	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
	{
		v1.x -= v2.x; v1.y -= v2.y;
		return v1;
	};

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator-(const vec2_t<t_id, T>& v1, const vec2_t<u_id, U>& v2)
		-> vec2_t<ASX::ID_value, decltype(v1.x - v2.x)>
	{
		vec2_t<ASX::ID_value, decltype(v1.x - v2.x)> temp(v1);
		return temp -= v2;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<t_id, T>& operator-=(vec2_t<t_id, T>& v, const U s)
	{
		v.x -= s; v.y -= s;
		return v;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator-(const T s, const vec2_t<t_id, U>& v)
		-> vec2_t<ASX::ID_value, decltype(s - v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s - v.x)> temp(v);
		return temp -= s;
	};

	template <ASX::ID t_id, class T, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto operator-(const vec2_t<t_id, T>& v, const U s)
		-> vec2_t<ASX::ID_value, decltype(s - v.x)>
	{
		vec2_t<ASX::ID_value, decltype(s - v.x)> temp(v);
		return temp -= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Dot Product                                */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	float dot(const vec4_t<t_id, T>& a, const vec4_t<u_id, U>& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);
	}

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	float dot(const vec3_t<t_id, T>& a, const vec3_t<u_id, U>& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
	}

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U > CLM_FUNC_DECL CLM_CONSTEXPR
	float dot(const vec2_t<t_id, T>& a, const vec2_t<u_id, U>& b)
	{
		return (a.x * b.x) + (a.y * b.y);
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> operator-(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(-v.x, -v.y, -v.z, -v.w);
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> operator-(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(-v.x, -v.y, -v.z);
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> operator-(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(-v.x, -v.y);
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> floor(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(::floor(v.x), ::floor(v.y), ::floor(v.z), ::floor(v.w));
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> floor(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(::floor(v.x), ::floor(v.y), ::floor(v.z));
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> floor(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(::floor(v.x), ::floor(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> ceil(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(::ceil(v.x), ::ceil(v.y), ::ceil(v.z), ::ceil(v.w));
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> ceil(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(::ceil(v.x), ::ceil(v.y), ::ceil(v.z));
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> ceil(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(::ceil(v.x), ::ceil(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> frac(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(frac(v.x), frac(v.y), frac(v.z), frac(v.w));
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> frac(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(frac(v.x), frac(v.y), frac(v.z));
	}

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> frac(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(frac(v.x), frac(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Abs                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> abs(const vec4_t<t_id, T>& v)
	{
		return vec4_t<ASX::ID_value, T>(::abs(v.x), ::abs(v.y), ::abs(v.z), ::abs(v.w));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> abs(const vec3_t<t_id, T>& v)
	{
		return vec3_t<ASX::ID_value, T>(::abs(v.x), ::abs(v.y), ::abs(v.z));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> abs(const vec2_t<t_id, T>& v)
	{
		return vec2_t<ASX::ID_value, T>(::abs(v.x), ::abs(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> clamp(const vec4_t<t_id, T>& v, const vec4_t<u_id, T>& min, const vec4_t<v_id, T>& max)
	{
		return vec4_t<ASX::ID_value, T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y), clamp(v.z, min.z, max.z), clamp(v.w, min.w, max.w));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> clamp(const vec4_t<t_id, T>& v, const T min, const T max)
	{
		return vec4_t<ASX::ID_value, T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max), clamp(v.w, min, max));
	}

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> clamp(const vec3_t<t_id, T>& v, const vec3_t<u_id, T>& min, const vec3_t<v_id, T>& max)
	{
		return vec3_t<ASX::ID_value, T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y), clamp(v.z, min.z, max.z));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> clamp(const vec3_t<t_id, T>& v, const T min, const T max)
	{
		return vec3_t<ASX::ID_value, T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max));
	}

	template <ASX::ID t_id, ASX::ID u_id, ASX::ID v_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> clamp(const vec2_t<t_id, T>& v, const vec2_t<u_id, T>& min, const vec2_t<v_id, T>& max)
	{
		return vec2_t<ASX::ID_value, T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> clamp(const vec2_t<t_id, T>& v, const T min, const T max)
	{
		return vec2_t<ASX::ID_value, T>(clamp(v.x, min, max), clamp(v.y, min, max));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Max                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> max(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(::fmax(x.x, y.x), ::fmax(x.y, y.y), ::fmax(x.z, y.z), ::fmax(x.w, y.w));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> max(const T x, const vec4_t<t_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(::fmax(x, y.x), ::fmax(x, y.y), ::fmax(x, y.z), ::fmax(x, y.w));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> max(const vec4_t<t_id, T>& x, const T y)
	{
		return vec4_t<ASX::ID_value, T>(::fmax(x.x, y), ::fmax(x.y, y), ::fmax(x.z, y), ::fmax(x.w, y));
	}

	template <ASX::ID t_id, ASX::ID u_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> max(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(::fmax(x.x, y.x), ::fmax(x.y, y.y), ::fmax(x.z, y.z));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> max(const T x, const vec3_t<t_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(::fmax(x, y.x), ::fmax(x, y.y), ::fmax(x, y.z));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> max(const vec3_t<t_id, T>& x, const T y)
	{
		return vec3_t<ASX::ID_value, T>(::fmax(x.x, y), ::fmax(x.y, y), ::fmax(x.z, y));
	}

	template <ASX::ID t_id, ASX::ID u_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> max(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(::fmax(x.x, y.x), ::fmax(x.y, y.y));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> max(const T x, const vec2_t<t_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(::fmax(x, y.x), ::fmax(x, y.y));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> max(const vec2_t<t_id, T>& x, const T y)
	{
		return vec2_t<ASX::ID_value, T>(::fmax(x.x, y), ::fmax(x.y, y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Min                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, ASX::ID u_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> min(const vec4_t<t_id, T>& x, const vec4_t<u_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(::fmin(x.x, y.x), ::fmin(x.y, y.y), ::fmin(x.z, y.z), ::fmin(x.w, y.w));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> min(const T x, const vec4_t<t_id, T>& y)
	{
		return vec4_t<ASX::ID_value, T>(::fmin(x, y.x), ::fmin(x, y.y), ::fmin(x, y.z), ::fmin(x, y.w));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> min(const vec4_t<t_id, T>& x, const T y)
	{
		return vec4_t<ASX::ID_value, T>(::fmin(x.x, y), ::fmin(x.y, y), ::fmin(x.z, y), ::fmin(x.w, y));
	}

	template <ASX::ID t_id, ASX::ID u_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> min(const vec3_t<t_id, T>& x, const vec3_t<u_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(::fmin(x.x, y.x), ::fmin(x.y, y.y), ::fmin(x.z, y.z));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> min(const T x, const vec3_t<t_id, T>& y)
	{
		return vec3_t<ASX::ID_value, T>(::fmin(x, y.x), ::fmin(x, y.y), ::fmin(x, y.z));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> min(const vec3_t<t_id, T>& x, const T y)
	{
		return vec3_t<ASX::ID_value, T>(::fmin(x.x, y), ::fmin(x.y, y), ::fmin(x.z, y));
	}

	template <ASX::ID t_id, ASX::ID u_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> min(const vec2_t<t_id, T>& x, const vec2_t<u_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(::fmin(x.x, y.x), ::fmin(x.y, y.y));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> min(const T x, const vec2_t<t_id, T>& y)
	{
		return vec2_t<ASX::ID_value, T>(::fmin(x, y.x), ::fmin(x, y.y));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> min(const vec2_t<t_id, T>& x, const T y)
	{
		return vec2_t<ASX::ID_value, T>(::fmin(x.x, y), ::fmin(x.y, y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec4_t<ASX::ID_value, T> pow(const vec4_t<t_id, T>& v, const T p)
	{
		return vec4_t<ASX::ID_value, T>(::pow(v.x, p), ::pow(v.y, p), ::pow(v.z, p), ::pow(v.w, p));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> pow(const vec3_t<t_id, T>& v, const T p)
	{
		return vec3_t<ASX::ID_value, T>(::pow(v.x, p), ::pow(v.y, p), ::pow(v.z, p));
	}
	template <ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec2_t<ASX::ID_value, T> pow(const vec2_t<t_id, T>& v, const T p)
	{
		return vec2_t<ASX::ID_value, T>(::pow(v.x, p), ::pow(v.y, p));
	}

	/* -------------------------------------------------------------------------- */
	/*                         Length, Distance, Normalize                        */
	/* -------------------------------------------------------------------------- */

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	float length(const vec_t<t_id, T>& v)
	{
		return ::sqrtf(dot(v, v));
	}

	template <template<ASX::ID, class> class vec_t, ASX::ID t_id, class T> CLM_FUNC_DECL CLM_CONSTEXPR
	vec_t<ASX::ID_value, T> normalize(const vec_t<t_id, T>& v)
	{
		vec_t<ASX::ID_value, T> tmp(v);
		float l = length(v);
		return (l == 0.f) ? tmp : tmp / l;
	}

	template <
		template<ASX::ID, class> class vec_t,
		ASX::ID t_id, class T,
		ASX::ID u_id, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	float distance(const vec_t<t_id, T>& a, const vec_t<u_id, U>& b)
	{
		return length(b - a);
	}

	template <
		template<ASX::ID, class> class vec_t,
		ASX::ID t_id, class T,
		ASX::ID u_id, class U,
		ASX::ID v_id, class V, 
		ASX::ID w_id, class W, 
		ASX::ID x_id, class X> CLM_FUNC_DECL CLM_CONSTEXPR
	auto remap(const vec_t<t_id, T>& h1, const vec_t<u_id, U>& l1, const vec_t<v_id, V>& h2, const vec_t<w_id, W>& l2, const vec_t<x_id, X>& v)
		-> vec_t<ASX::ID_value, decltype(l2.x + (v.x - l1.x) * (h2.x - l2.x) / (h1.x - l1.x))>
	{
		return l2 + (v - l1) * (h2 - l2) / (h1 - l1);
	}

	template <
		template<ASX::ID, class> class vec_t,
		ASX::ID t_id, class T,
		ASX::ID u_id, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto mix(const vec_t<t_id, T>&a, const vec_t<u_id, U>&b, const float t)
		-> vec_t<ASX::ID_value, decltype(a.x * (1.0 - t) + b.x * t)>
	{
		return a * (1.f - t) + b * t;
	}

	template <
		template<ASX::ID, class> class vec_t,
		ASX::ID t_id, class T,
		ASX::ID u_id, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	auto smooth_step(const vec_t<t_id, T>& a, const vec_t<u_id, U>& b, const float x)
		-> vec_t<ASX::ID_value, decltype((x - a.x) / (b.x - a.x))>
	{
		float y = clamp((x - a) / (b - a), 0.f, 1.f);
		return (y * y * (3.f - (2.f * y)));
	}

	/* -------------------------------------------------------------------------- */
	/*                               Cross, Reflect                               */
	/* -------------------------------------------------------------------------- */

	template <
		ASX::ID t_id, class T, 
		ASX::ID u_id, class U> CLM_FUNC_DECL CLM_CONSTEXPR
	vec3_t<ASX::ID_value, T> reflect(const vec3_t<t_id, T>& i, const vec3_t<u_id, U>& n)
	{
		return i - 2.f * n * dot(n, i);
	}

	template <
		ASX::ID t_id, class T,
		ASX::ID u_id, class U> CLM_FUNC_DECL CLM_CONSTEXPR
		vec3_t<ASX::ID_value, T> cross(const vec3_t<t_id, T>& a, const vec3_t<u_id, U>& b)
	{
		return vec3_t<ASX::ID_value, T>(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
	}

	template<class T> CLM_FUNC_DECL CLM_CONSTEXPR
		T luminance(const T r, const T g, const T b)
	{
		return (0.299 * r + 0.587 * g + 0.114 * b);
	}
	static CLM_FUNC_DECL
		float luminance(const vec3f& color)
	{
		return (0.299 * color.x + 0.587 * color.y + 0.114 * color.z);
	}

	template <ASX::ID t_id, class T>
	static CLM_FUNC_DECL vec3<T> gram_schmidt(const vec3_t<t_id, T>& v)
	{
		vec3<T> x(rand_float(-1, 1), rand_float(-1, 1), rand_float(-1, 1));

		float x_dot_v = dot(x, v);

		vec3<T> v_norm = normalize(v);
		vec3<T> v_norm_2 = v_norm * v_norm;

		x = x - ((x_dot_v * v) / v_norm_2);

		return normalize(x);
	}
	static CLM_FUNC_DECL
		vec2f sample_spherical_map(const vec3f& d)
	{
		vec2f uv = vec2f(0.5f + ::atan2(d.z, d.x) * M_1_2PI, 0.5f - ::asin(d.y) * M_1_PI);
		return uv;
	}
	static CLM_FUNC_DECL
		vec3f sample_spherical_direction(const vec2f& uv)
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

#endif // _CML_VECTOR_