#ifndef _JEK_VECTOR_
#define _JEK_VECTOR_

#include "CudaCommon.h"
#include "GLCommon.h"
#include "Random.h"
#include "dMath.h"

namespace jek
{
	template <class T> struct Vec4;
	template <class T> struct Vec3;
	template <class T> struct Vec2;

	template <class T> struct _ALIGN(16) Vec4
	{
		static_assert(sizeof(T) == 4, "Type must be 4 bytes");
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		T x{}, y{}, z{}, w{};
		_HOST_DEVICE Vec4() {};
		_HOST_DEVICE Vec4(T x, T y, T z, T w) : x(x), y(y), z(z), w(w) {};
		_HOST_DEVICE Vec4(T s) : x(s), y(s), z(s), w(s) {};
		_HOST_DEVICE Vec4(const Vec4<T>&v) : x(v.x), y(v.y), z(v.z), w(v.w) {};
		_HOST_DEVICE Vec4(const Vec3<T>&v, T w) : x(v.x), y(v.y), z(v.z), w(w) {};
		_HOST_DEVICE Vec4(const Vec3<T>&v) : x(v.x), y(v.y), z(v.z), w(0) {};
		_HOST_DEVICE Vec4(const Vec2<T>&v, T z, T w) : x(v.x), y(v.y), z(z), w(w) {};
		_HOST_DEVICE Vec4(const Vec2<T>&v) : x(v.x), y(v.y), z(0), w(0) {};

		_HOST_DEVICE Vec4(const float4 & v) : x(v.x), y(v.y), z(v.z), w(v.w) {};
		_HOST_DEVICE Vec4(const float3 & v) : x(v.x), y(v.y), z(v.z), w(0) {};
		_HOST_DEVICE Vec4(const float2 & v) : x(v.x), y(v.y), z(0), w(0) {};

		_HOST Vec4(const glm::vec4 & v) : x(v.x), y(v.y), z(v.z), w(v.w) {};
		_HOST Vec4(const glm::vec3 & v) : x(v.x), y(v.y), z(v.z), w(0) {};
		_HOST Vec4(const glm::vec2 & v) : x(v.x), y(v.y), z(0), w(0) {};

		template <class U>
		_HOST_DEVICE operator Vec4<U>() const
		{
			return Vec4<U>(x, y, z, w);
		};
		template <class U>
		_HOST_DEVICE operator Vec3<U>() const
		{
			return Vec3<U>(x, y, z);
		};
		template <class U>
		_HOST_DEVICE operator Vec2<U>() const
		{
			return Vec2<U>(x, y);
		};

		_HOST_DEVICE operator float4() const
		{
			return make_float4(x, y, z, w);
		};
		_HOST_DEVICE operator float3() const
		{
			return make_float3(x, y, z);
		};
		_HOST_DEVICE operator float2() const
		{
			return make_float2(x, y);
		};

		_HOST operator glm::vec4() const
		{
			return glm::vec4(x, y, z, w);
		};
		_HOST operator glm::vec3() const
		{
			return glm::vec3(x, y, z);
		};
		_HOST operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};

		_HOST_DEVICE void print() const
		{
			printf("(%f, %f, %f, %f)\n", (float)x, (float)y, (float)z, (float)w);
		};
	};
	template <class T> struct _ALIGN(16) Vec3
	{
		//static_assert(sizeof(T) == 4, "Type must be 4 bytes");
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		T x{}, y{}, z{};
		_HOST_DEVICE Vec3() {};
		_HOST_DEVICE Vec3(T x, T y, T z) : x(x), y(y), z(z) {};
		_HOST_DEVICE Vec3(T s) : x(s), y(s), z(s) {};
		_HOST_DEVICE Vec3(const Vec4<T>&v) : x(v.x), y(v.y), z(v.z) {};
		_HOST_DEVICE Vec3(const Vec3<T>&v) : x(v.x), y(v.y), z(v.z) {};
		_HOST_DEVICE Vec3(const Vec2<T>&v, T z) : x(v.x), y(v.y), z(z) {};
		_HOST_DEVICE Vec3(const Vec2<T>&v) : x(v.x), y(v.y), z(0) {};

		_HOST_DEVICE Vec3(const float4 & v) : x(v.x), y(v.y), z(v.z) {};
		_HOST_DEVICE Vec3(const float3 & v) : x(v.x), y(v.y), z(v.z) {};
		_HOST_DEVICE Vec3(const float2 & v) : x(v.x), y(v.y), z(0) {};

		_HOST Vec3(const glm::vec4 & v) : x(v.x), y(v.y), z(v.z) {};
		_HOST Vec3(const glm::vec3 & v) : x(v.x), y(v.y), z(v.z) {};
		_HOST Vec3(const glm::vec2 & v) : x(v.x), y(v.y), z(0) {};

		template <class U>
		_HOST_DEVICE operator Vec4<U>() const
		{
			return Vec4<U>(x, y, z, 0);
		};
		template <class U>
		_HOST_DEVICE operator Vec3<U>() const
		{
			return Vec3<U>(x, y, z);
		};
		template <class U>
		_HOST_DEVICE operator Vec2<U>() const
		{
			return Vec2<U>(x, y);
		};

		_HOST_DEVICE operator float4() const
		{
			return make_float4(x, y, z, 0.f);
		};
		_HOST_DEVICE operator float3() const
		{
			return make_float3(x, y, z);
		};
		_HOST_DEVICE operator float2() const
		{
			return make_float2(x, y);
		};

		_HOST operator glm::vec4() const
		{
			return glm::vec4(x, y, z, 0);
		};
		_HOST operator glm::vec3() const
		{
			return glm::vec3(x, y, z);
		};
		_HOST operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};

		_HOST_DEVICE void print() const
		{
			printf("(%f, %f, %f)\n", (float)x, (float)y, (float)z);
		};
	private:
		T padd{};
	};
	template <class T> struct _ALIGN(8) Vec2
	{
		static_assert(sizeof(T) == 4, "Type must be 4 bytes");
		static_assert(std::is_arithmetic<T>::value, "Type must be must be numeric");

		T x{}, y{};
		_HOST_DEVICE Vec2() {};
		_HOST_DEVICE Vec2(T x, T y) : x(x), y(y) {};
		_HOST_DEVICE Vec2(T s) : x(s), y(s) {};
		_HOST_DEVICE Vec2(const Vec4<T>&v) : x(v.x), y(v.y) {};
		_HOST_DEVICE Vec2(const Vec3<T>&v) : x(v.x), y(v.y) {};
		_HOST_DEVICE Vec2(const Vec2<T>&v) : x(v.x), y(v.y) {};

		_HOST_DEVICE Vec2(const float4 & v) : x(v.x), y(v.y) {};
		_HOST_DEVICE Vec2(const float3 & v) : x(v.x), y(v.y) {};
		_HOST_DEVICE Vec2(const float2 & v) : x(v.x), y(v.y) {};

		_HOST Vec2(const glm::vec4 & v) : x(v.x), y(v.y) {};
		_HOST Vec2(const glm::vec3 & v) : x(v.x), y(v.y) {};
		_HOST Vec2(const glm::vec2 & v) : x(v.x), y(v.y) {};

		template <class U>
		_HOST_DEVICE operator Vec4<U>() const
		{
			return Vec4<U>(x, y, 0, 0);
		};
		template <class U>
		_HOST_DEVICE operator Vec3<U>() const
		{
			return Vec3<U>(x, y, 0);
		};
		template <class U>
		_HOST_DEVICE operator Vec2<U>() const
		{
			return Vec2<U>(x, y);
		};

		_HOST_DEVICE operator float4() const
		{
			return make_float4(x, y, 0.f, 0.f);
		};
		_HOST_DEVICE operator float3() const
		{
			return make_float3(x, y, 0.f);
		};
		_HOST_DEVICE operator float2() const
		{
			return make_float2(x, y);
		};

		_HOST operator glm::vec4() const
		{
			return glm::vec4(x, y, 0, 0);
		};
		_HOST operator glm::vec3() const
		{
			return glm::vec3(x, y, 0);
		};
		_HOST operator glm::vec2() const
		{
			return glm::vec2(x, y);
		};

		_HOST_DEVICE void print() const
		{
			printf("(%f, %f)\n", (float)x, (float)y);
		};
	};

	typedef Vec4<float>		Vec4f;
	typedef Vec4<int32_t>	Vec4i;
	typedef Vec4<uint32_t>	Vec4u;

	typedef Vec3<float>		Vec3f;
	typedef Vec3<int32_t>	Vec3i;
	typedef Vec3<uint32_t>	Vec3u;

	typedef Vec2<float>		Vec2f;
	typedef Vec2<int32_t>	Vec2i;
	typedef Vec2<uint32_t>	Vec2u;

	/*
	* Ignore this crap. Just testing out some container stuff.
	* 
	* 
	struct SoA_Container {
		SoA_Container& operator[](int index) 
		{
			return reinterpret_cast<SoA_Container&>(comp1[index]);
		};
		union { float comp1val, comp1[100]; };
		union { float comp2val, comp2[100]; };
		union { float comp3val, comp3[100]; };
	};

	template <class T> struct SOA_Vec3
	{
		class ProxyVec
		{
			//SOA_Vec3& soa;
			T* x;
			T* y;
			T* z;
			int idx;
		public:
			_HOST_DEVICE ProxyVec(T** x, T** y, T** z, int idx) : x(*x), y(*y), z(*z), idx(idx) {};
			_HOST_DEVICE void operator=(ProxyVec& v)
			{
				auto& _idx = v.idx;

				x[idx] = v.x[_idx];
				y[idx] = v.y[_idx];
				z[idx] = v.z[_idx];
			};
			_HOST_DEVICE ProxyVec& operator+=(const ProxyVec& v)
			{
				const auto& _idx = v.idx;

				x[idx] += v.x[_idx];
				y[idx] += v.y[_idx];
				z[idx] += v.z[_idx];

				return *this;
			};
			_HOST_DEVICE ProxyVec& operator+(ProxyVec& v)
			{
				return *this += v;
			};

			_HOST_DEVICE operator Vec3<T>() const { return { x[idx], y[idx], z[idx] }; };
		};

		_HOST_DEVICE ProxyVec operator[] (int index) { return ProxyVec(&x, &y, &z, index); }
		
		~SOA_Vec3()
		{
			//cudaFree(x);
			//cudaFree(y);
			//cudaFree(z);
			//cudaFree(this*);
		};

		_HOST [[nodiscard]] static SOA_Vec3<T> create(std::vector<Vec3<T>> vec_array)
		{
			auto size = vec_array.size();

			T* x = new T[size];
			T* y = new T[size];
			T* z = new T[size];

			for (int i = 0; i < size; i++)
			{
				x[i] = vec_array[i].x;
				y[i] = vec_array[i].y;
				z[i] = vec_array[i].z;
			}

			SOA_Vec3<T> out_ptr;

			auto alloc_size = sizeof(T) * size;

			//cudaMalloc((void**)&out_ptr, sizeof(SOA_Vec3<T>));
			cudaMalloc((void**)&out_ptr.x, alloc_size);
			cudaMalloc((void**)&out_ptr.y, alloc_size);
			cudaMalloc((void**)&out_ptr.z, alloc_size);

			cudaMemcpy((void**)out_ptr.x, x, alloc_size, cudaMemcpyHostToDevice);
			cudaMemcpy((void**)out_ptr.y, y, alloc_size, cudaMemcpyHostToDevice);
			cudaMemcpy((void**)out_ptr.z, z, alloc_size, cudaMemcpyHostToDevice);

			//out_ptr->size = size;
			return out_ptr;
		};

		
		_HOST_DEVICE Vec3<T> operator[](const int idx) const
		{
			//(this)->idx = idx;
			return { x[idx], y[idx], z[idx] };
		};
		_HOST_DEVICE void set(const int idx, const Vec3<T> v)
		{
			x[idx] = v.x;
			y[idx] = v.y;
			z[idx] = v.z;
		};
		
		/*
		_HOST_DEVICE Vec3<T>& operator[](const int index) 
		{
			(this)->idx = idx;
			auto out = Vec3<T>(x[idx], y[idx], z[idx]);
			return &out;
		};

		_HOST_DEVICE void operator=(const Vec3<T>& v)
		{
			x[idx] = v.x;
			y[idx] = v.y;
			z[idx] = v.z;
		};
		
		//unsigned long operator [](int i) const { return registers[i]; }
		//unsigned long& operator [](int i) { return registers[i]; }

		_HOST_DEVICE void get_arrays(T** x, T** y, T** z)
		{
			*x = (this)->x;
			*y = (this)->y;
			*z = (this)->z;
		}

		_HOST_DEVICE T* get_x()
		{
			return x;
		}
		_HOST_DEVICE T* get_y()
		{
			return y;
		}
		_HOST_DEVICE T* get_z()
		{
			return z;
		}

	private:
		SOA_Vec3() {}

		T* x;
		T* y;
		T* z;
		uint32_t size;
		uint32_t idx = 0;
	};

	typedef SOA_Vec3<float>		SOA_Vec3f;
	typedef SOA_Vec3<int32_t>	SOA_Vec3i;
	typedef SOA_Vec3<uint32_t>	SOA_Vec3u;
	*/

	/* -------------------------------------------------------------------------- */
	/*                                 Comparators                                */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline bool operator==(const Vec4<T>& v1, const Vec4<U>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z && v1.w == v2.w);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator!=(const Vec4<T>& v1, const Vec4<U>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z || v1.w != v2.w);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator<(const Vec4<T>& v1, const Vec4<U>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y&& v1.z < v2.z&& v1.w < v2.w);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator<=(const Vec4<T>& v1, const Vec4<U>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z && v1.w <= v2.w);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator>(const Vec4<T>& v1, const Vec4<U>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z && v1.w > v2.w);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator>=(const Vec4<T>& v1, const Vec4<U>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z && v1.w >= v2.w);
	};

	template <class T, class U> _HOST_DEVICE
		inline bool operator==(const Vec3<T>& v1, const Vec3<U>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y && v1.z == v2.z);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator!=(const Vec3<T>& v1, const Vec3<U>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y || v1.z != v2.z);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator<(const Vec3<T>& v1, const Vec3<U>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y&& v1.z < v2.z);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator<=(const Vec3<T>& v1, const Vec3<U>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y && v1.z <= v2.z);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator>(const Vec3<T>& v1, const Vec3<U>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y && v1.z > v2.z);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator>=(const Vec3<T>& v1, const Vec3<U>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y && v1.z >= v2.z);
	};

	template <class T, class U> _HOST_DEVICE
		inline bool operator==(const Vec2<T>& v1, const Vec2<U>& v2)
	{
		return (v1.x == v2.x && v1.y == v2.y);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator!=(const Vec2<T>& v1, const Vec2<U>& v2)
	{
		return (v1.x != v2.x || v1.y != v2.y);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator<(const Vec2<T>& v1, const Vec2<U>& v2)
	{
		return (v1.x < v2.x&& v1.y < v2.y);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator<=(const Vec2<T>& v1, const Vec2<U>& v2)
	{
		return (v1.x <= v2.x && v1.y <= v2.y);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator>(const Vec2<T>& v1, const Vec2<U>& v2)
	{
		return (v1.x > v2.x && v1.y > v2.y);
	};
	template <class T, class U> _HOST_DEVICE
		inline bool operator>=(const Vec2<T>& v1, const Vec2<U>& v2)
	{
		return (v1.x >= v2.x && v1.y >= v2.y);
	};

	/* -------------------------------------------------------------------------- */
	/*                               Multiplication                               */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x* v2.x)>
	{
		v1 = { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x* s)>
	{
		v = { s * v.x, s * v.y, s * v.z, s * v.w };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x* v2.x)>
	{
		auto temp(v1);
		return temp *= v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const T s, const Vec4<U>& v)
		-> Vec4<decltype(s* v.x)>
	{
		auto temp(v);
		return temp *= s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x* s)>
	{
		auto temp(v);
		return temp *= s;
	};

	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x* v2.x)>
	{
		v1 = { v1.x * v2.x, v1.y * v2.y, v1.z * v2.z };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x* s)>
	{
		v = { s * v.x, s * v.y, s * v.z };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x* v2.x)>
	{
		auto temp(v1);
		return temp *= v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const T s, const Vec3<U>& v)
		-> Vec3<decltype(s* v.x)>
	{
		auto temp(v);
		return temp *= s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x* s)>
	{
		auto temp(v);
		return temp *= s;
	};

	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x* v2.x)>
	{
		v1 = { v1.x * v2.x, v1.y * v2.y };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*=(Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x* s)>
	{
		v = { s * v.x, s * v.y };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x* v2.x)>
	{
		auto temp(v1);
		return temp *= v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const T s, const Vec2<U>& v)
		-> Vec2<decltype(s* v.x)>
	{
		auto temp(v);
		return temp *= s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator*(const Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x* s)>
	{
		auto temp(v);
		return temp *= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Division                                  */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x / v2.x)>
	{
		v1 = { v1.x / v2.x, v1.y / v2.y, v1.z / v2.z, v1.w / v2.w };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x / s)>
	{
		v = { v.x / s, v.y / s, v.z / s, v.w / s };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x / v2.x)>
	{
		auto temp(v1);
		return temp /= v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const T s, const Vec4<U>& v)
		-> Vec4<decltype(s / v.x)>
	{
		Vec4<decltype(s / v.x)>temp(s);
		return temp /= v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x / s)>
	{
		auto temp(v);
		return temp /= s;
	};

	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x / v2.x)>
	{
		v1 = { v1.x / v2.x, v1.y / v2.y, v1.z / v2.z };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x / s)>
	{
		v = { v.x / s, v.y / s, v.z / s };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x / v2.x)>
	{
		auto temp(v1);
		return temp /= v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const T s, const Vec3<U>& v)
		-> Vec3<decltype(s / v.x)>
	{
		Vec3<decltype(s / v.x)>temp(s);
		return temp /= v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x / s)>
	{
		auto temp(v);
		return temp /= s;
	};

	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x / v2.x)>
	{
		v1 = { v1.x / v2.x, v1.y / v2.y };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/=(Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x / s)>
	{
		v = { v.x / s, v.y / s };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x / v2.x)>
	{
		auto temp(v1);
		return temp /= v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const T s, const Vec2<U>& v)
		-> Vec2<decltype(s / v.x)>
	{
		Vec2<decltype(s / v.x)>temp(s);
		return temp /= v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator/(const Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x / s)>
	{
		auto temp(v);
		return temp /= s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                  Addition                                  */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x + v2.x)>
	{
		v1 = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x + s)>
	{
		v = { s + v.x, s + v.y, s + v.z, s + v.w };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x + v2.x)>
	{
		auto temp(v1);
		return temp += v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const T s, const Vec4<U>& v)
		-> Vec4<decltype(s + v.x)>
	{
		auto temp(v);
		return temp += s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x + s)>
	{
		auto temp(v);
		return temp += s;
	};

	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x + v2.x)>
	{
		v1 = { v1.x + v2.x, v1.y + v2.y, v1.z + v2.z};
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x + s)>
	{
		v = { s + v.x, s + v.y, s + v.z };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x + v2.x)>
	{
		auto temp(v1);
		return temp += v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const T s, const Vec3<U>& v)
		-> Vec3<decltype(s + v.x)>
	{
		auto temp(v);
		return temp += s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x + s)>
	{
		auto temp(v);
		return temp += s;
	};

	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x + v2.x)>
	{
		v1 = { v1.x + v2.x, v1.y + v2.y };
		return v1;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+=(Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x + s)>
	{
		v = { s + v.x, s + v.y };
		return v;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x + v2.x)>
	{
		auto temp(v1);
		return temp += v2;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const T s, const Vec2<U>& v)
		-> Vec2<decltype(s + v.x)>
	{
		auto temp(v);
		return temp += s;
	};
	template <class T, class U> _HOST_DEVICE
		inline auto operator+(const Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x + s)>
	{
		auto temp(v);
		return temp += s;
	};

	/* -------------------------------------------------------------------------- */
	/*                                 Subtraction                                */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x - v2.x)>
	{
		v1 = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.w - v2.w };
		return v1;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x - s)>
	{
		v = { v.x - s, v.y - s, v.z - s, v.w - s };
		return v;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Vec4<T>& v1, const Vec4<U>& v2)
		-> Vec4<decltype(v1.x - v2.x)>
	{
		auto temp(v1);
		return temp -= v2;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const T s, const Vec4<U>& v)
		-> Vec4<decltype(s - v.x)>
	{
		Vec4<decltype(s - v.x)>temp(s);
		return temp -= v;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Vec4<T>& v, const U s)
		-> Vec4<decltype(v.x - s)>
	{
		auto temp(v);
		return temp -= s;
	}

	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x - v2.x)>
	{
		v1 = { v1.x - v2.x, v1.y - v2.y, v1.z - v2.z };
		return v1;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x - s)>
	{
		v = { v.x - s, v.y - s, v.z - s };
		return v;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Vec3<T>& v1, const Vec3<U>& v2)
		-> Vec3<decltype(v1.x - v2.x)>
	{
		auto temp(v1);
		return temp -= v2;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const T s, const Vec3<U>& v)
		-> Vec3<decltype(s - v.x)>
	{
		Vec3<decltype(s - v.x)>temp(s);
		return temp -= v;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Vec3<T>& v, const U s)
		-> Vec3<decltype(v.x - s)>
	{
		auto temp(v);
		return temp -= s;
	}

	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x - v2.x)>
	{
		v1 = { v1.x - v2.x, v1.y - v2.y };
		return v1;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-=(Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x - s)>
	{
		v = { v.x - s, v.y - s };
		return v;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Vec2<T>& v1, const Vec2<U>& v2)
		-> Vec2<decltype(v1.x - v2.x)>
	{
		auto temp(v1);
		return temp -= v2;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const T s, const Vec2<U>& v)
		-> Vec2<decltype(s - v.x)>
	{
		Vec2<decltype(s - v.x)>temp(s);
		return temp -= v;
	}
	template <class T, class U> _HOST_DEVICE
		inline auto operator-(const Vec2<T>& v, const U s)
		-> Vec2<decltype(v.x - s)>
	{
		auto temp(v);
		return temp -= s;
	}

	/* -------------------------------------------------------------------------- */
	/*                                 Dot Product                                */
	/* -------------------------------------------------------------------------- */

	template <class T, class U> _HOST_DEVICE
		inline float dot(const Vec4<T>& a, const Vec4<U>& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z) + (a.w * b.w);
	}
	template <class T, class U> _HOST_DEVICE
		inline float dot(const Vec3<T>& a, const Vec3<U>& b)
	{
		return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
	}
	template <class T, class U> _HOST_DEVICE
		inline float dot(const Vec2<T>& a, const Vec2<U>& b)
	{
		return (a.x * b.x) + (a.y * b.y);
	}

	/* -------------------------------------------------------------------------- */
	/*                                  Negation                                  */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> operator-(const Vec4<T>& v)
	{
		return Vec4<T>(-v.x, -v.y, -v.z, -v.w);
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> operator-(const Vec3<T>& v)
	{
		return Vec3<T>(-v.x, -v.y, -v.z);
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> operator-(const Vec2<T>& v)
	{
		return Vec2<T>(-v.x, -v.y);
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Floor                                   */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> floor(const Vec4<T>& v)
	{
		return Vec4<T>(::floor(v.x), ::floor(v.y), ::floor(v.z), ::floor(v.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> floor(const Vec3<T>& v)
	{
		return Vec3<T>(::floor(v.x), ::floor(v.y), ::floor(v.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> floor(const Vec2<T>& v)
	{
		return Vec2<T>(::floor(v.x), ::floor(v.y));
	}


	/* -------------------------------------------------------------------------- */
	/*                                    Ceil                                    */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> ceil(const Vec4<T>& v)
	{
		return Vec4<T>(::ceil(v.x), ::ceil(v.y), ::ceil(v.z), ::ceil(v.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> ceil(const Vec3<T>& v)
	{
		return Vec3<T>(::ceil(v.x), ::ceil(v.y), ::ceil(v.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> ceil(const Vec2<T>& v)
	{
		return Vec2<T>(::ceil(v.x), ::ceil(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Frac                                    */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> frac(const Vec4<T>& v)
	{
		return Vec4<T>(frac(v.x), frac(v.y), frac(v.z), frac(v.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> frac(const Vec3<T>& v)
	{
		return Vec3<T>(frac(v.x), frac(v.y), frac(v.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> frac(const Vec2<T>& v)
	{
		return Vec2<T>(frac(v.x), frac(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Abs                                    */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> abs(const Vec4<T>& v)
	{
		return Vec4<T>(::abs(v.x), ::abs(v.y), ::abs(v.z), ::abs(v.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> abs(const Vec3<T>& v)
	{
		return Vec3<T>(::abs(v.x), ::abs(v.y), ::abs(v.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> abs(const Vec2<T>& v)
	{
		return Vec2<T>(::abs(v.x), ::abs(v.y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                    Clamp                                   */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> clamp(const Vec4<T>& v, const Vec4<T>& min, const Vec4<T>& max)
	{
		return Vec4<T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y), clamp(v.z, min.z, max.z), clamp(v.w, min.w, max.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec4<T> clamp(const Vec4<T>& v, const T min, const T max)
	{
		return Vec4<T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max), clamp(v.w, min, max));
	}

	template<class T> _HOST_DEVICE
		inline Vec3<T> clamp(const Vec3<T>& v, const Vec3<T>& min, const Vec3<T>& max)
	{
		return Vec3<T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y), clamp(v.z, min.z, max.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> clamp(const Vec3<T>& v, const T min, const T max)
	{
		return Vec3<T>(clamp(v.x, min, max), clamp(v.y, min, max), clamp(v.z, min, max));
	}

	template<class T> _HOST_DEVICE
		inline Vec2<T> clamp(const Vec2<T>& v, const Vec2<T>& min, const Vec2<T>& max)
	{
		return Vec2<T>(clamp(v.x, min.x, max.x), clamp(v.y, min.y, max.y));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> clamp(const Vec2<T>& v, const T min, const T max)
	{
		return Vec2<T>(clamp(v.x, min, max), clamp(v.y, min, max));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Max                                    */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> max(const Vec4<T>& x, const Vec4<T>& y)
	{
		return Vec4<T>(::fmax(x.x, y.x), ::fmax(x.y, y.y), ::fmax(x.z, y.z), ::fmax(x.w, y.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec4<T> max(const T x, const Vec4<T>& y)
	{
		return Vec4<T>(::fmax(x, y.x), ::fmax(x, y.y), ::fmax(x, y.z), ::fmax(x, y.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec4<T> max(const Vec4<T>& x, const T y)
	{
		return Vec4<T>(::fmax(x.x, y), ::fmax(x.y, y), ::fmax(x.z, y), ::fmax(x.w, y));
	}

	template<class T> _HOST_DEVICE
		inline Vec3<T> max(const Vec3<T>& x, const Vec3<T>& y)
	{
		return Vec3<T>(::fmax(x.x, y.x), ::fmax(x.y, y.y), ::fmax(x.z, y.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> max(const T x, const Vec3<T>& y)
	{
		return Vec3<T>(::fmax(x, y.x), ::fmax(x, y.y), ::fmax(x, y.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> max(const Vec3<T>& x, const T y)
	{
		return Vec3<T>(::fmax(x.x, y), ::fmax(x.y, y), ::fmax(x.z, y));
	}

	template<class T> _HOST_DEVICE
		inline Vec2<T> max(const Vec2<T>& x, const Vec2<T>& y)
	{
		return Vec2<T>(::fmax(x.x, y.x), ::fmax(x.y, y.y));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> max(const T x, const Vec2<T>& y)
	{
		return Vec2<T>(::fmax(x, y.x), ::fmax(x, y.y));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> max(const Vec2<T>& x, const T y)
	{
		return Vec2<T>(::fmax(x.x, y), ::fmax(x.y, y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Min                                    */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> min(const Vec4<T>& x, const Vec4<T>& y)
	{
		return Vec4<T>(::fmin(x.x, y.x), ::fmin(x.y, y.y), ::fmin(x.z, y.z), ::fmin(x.w, y.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec4<T> min(const T x, const Vec4<T>& y)
	{
		return Vec4<T>(::fmin(x, y.x), ::fmin(x, y.y), ::fmin(x, y.z), ::fmin(x, y.w));
	}
	template<class T> _HOST_DEVICE
		inline Vec4<T> min(const Vec4<T>& x, const T y)
	{
		return Vec4<T>(::fmin(x.x, y), ::fmin(x.y, y), ::fmin(x.z, y), ::fmin(x.w, y));
	}

	template<class T> _HOST_DEVICE
		inline Vec3<T> min(const Vec3<T>& x, const Vec3<T>& y)
	{
		return Vec3<T>(::fmin(x.x, y.x), ::fmin(x.y, y.y), ::fmin(x.z, y.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> min(const T x, const Vec3<T>& y)
	{
		return Vec3<T>(::fmin(x, y.x), ::fmin(x, y.y), ::fmin(x, y.z));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> min(const Vec3<T>& x, const T y)
	{
		return Vec3<T>(::fmin(x.x, y), ::fmin(x.y, y), ::fmin(x.z, y));
	}

	template<class T> _HOST_DEVICE
		inline Vec2<T> min(const Vec2<T>& x, const Vec2<T>& y)
	{
		return Vec2<T>(::fmin(x.x, y.x), ::fmin(x.y, y.y));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> min(const T x, const Vec2<T>& y)
	{
		return Vec2<T>(::fmin(x, y.x), ::fmin(x, y.y));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> min(const Vec2<T>& x, const T y)
	{
		return Vec2<T>(::fmin(x.x, y), ::fmin(x.y, y));
	}

	/* -------------------------------------------------------------------------- */
	/*                                     Pow                                    */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec4<T> pow(const Vec4<T>& v, const T p)
	{
		return Vec4<T>(::pow(v.x, p), ::pow(v.y, p), ::pow(v.z, p), ::pow(v.w, p));
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> pow(const Vec3<T>& v, const T p)
	{
		return Vec3<T>(::pow(v.x, p), ::pow(v.y, p), ::pow(v.z, p));
	}
	template<class T> _HOST_DEVICE
		inline Vec2<T> pow(const Vec2<T>& v, const T p)
	{
		return Vec2<T>(::pow(v.x, p), ::pow(v.y, p));
	}

	/* -------------------------------------------------------------------------- */
	/*                         Length, Distance, Normalize                        */
	/* -------------------------------------------------------------------------- */

	template <template<class> class VecType, class T> _HOST_DEVICE
		inline float length(const VecType<T>& v)
	{
		return ::sqrtf(dot(v, v));
	}
	template <template<class> class VecType, class T> _HOST_DEVICE
		inline float distance(const VecType<T>& a, const VecType<T>& b)
	{
		return length(b - a);
	}
	template <template<class> class VecType, class T> _HOST_DEVICE
		inline VecType<T> normalize(const VecType<T>& v)
	{
		float l = length(v);
		return (l == 0.f) ? v : v / l;
	}
	template <template<class> class VecType, class T> _HOST_DEVICE
		inline VecType<T> remap(const VecType<T>& h1, const VecType<T>& l1, const VecType<T>& h2, const VecType<T>& l2, const VecType<T>& v)
	{
		return l2 + (v - l1) * (h2 - l2) / (h1 - l1);
	}
	template <template<class> class VecType, class T> _HOST_DEVICE
		inline VecType<T> mix(const VecType<T>& a, const VecType<T>& b, const T t)
	{
		return a * ((T)1 - t) + b * t;
	}
	template <template<class> class VecType, class T> _HOST_DEVICE
		inline VecType<T> smooth_step(const VecType<T>& a, const VecType<T>& b, const T x)
	{
		T y = clamp((x - a) / (b - a), 0, 1);
		return (y * y * (T(3) - (T(2) * y)));
	}

	/* -------------------------------------------------------------------------- */
	/*                               Cross, Reflect                               */
	/* -------------------------------------------------------------------------- */

	template<class T> _HOST_DEVICE
		inline Vec3<T> reflect(const Vec3<T>& i, const Vec3<T>& n)
	{
		return i - 2.f * n * dot(n, i);
	}
	template<class T> _HOST_DEVICE
		inline Vec3<T> cross(const Vec3<T>& a, const Vec3<T>& b)
	{
		return Vec3<T>(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x);
	}

	template<class T> _HOST_DEVICE
		inline T luminance(const T r, const T g, const T b)
	{
		return (0.299 * r + 0.587 * g + 0.114 * b);
	}
	static inline _HOST_DEVICE
		float luminance(const Vec3f& color)
	{
		return (0.299 * color.x + 0.587 * color.y + 0.114 * color.z);
	}
	static inline _HOST_DEVICE
		Vec3f gram_schmidt(const Vec3f& v)
	{
		Vec3f x = Vec3f(rand_float(-1, 1), rand_float(-1, 1), rand_float(-1, 1));

		float x_dot_v = dot(x, v);

		Vec3f v_norm = normalize(v);
		Vec3f v_norm_2 = v_norm * v_norm;

		x = x - ((x_dot_v * v) / v_norm_2);

		return normalize(x);
	}
	static inline _HOST_DEVICE
		Vec2f sample_spherical_map(const Vec3f& d)
	{
		Vec2f uv = Vec2f(0.5f + ::atan2(d.z, d.x) * M_1_2PI, 0.5f - ::asin(d.y) * M_1_PI);
		return uv;
	}
	static inline _HOST_DEVICE
		Vec3f sample_spherical_direction(const Vec2f& uv)
	{
		float phi = 2.f * M_PI * (uv.x - 0.5f);
		float theta = M_PI * uv.y;

		Vec3f n;
		n.x = ::cos(phi) * ::sin(theta);
		n.z = ::sin(phi) * ::sin(theta);
		n.y = ::cos(theta);

		return n;
	}

} // namespace jek

#endif // _JEK_VECTOR_