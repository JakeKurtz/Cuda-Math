#ifndef _CML_MAT_
#define _CML_MAT_

#include "cuda_common.h"
#include "gl_common.h"
#include "vec.h"

namespace cml
{
    template <ASX::ID, typename T> struct mat4x4_t;
    template <ASX::ID, typename T> struct mat3x3_t;
    template <ASX::ID, typename T> struct mat2x2_t;

    template <typename T> using mat4x4 = mat4x4_t<ASX::ID_value, T>;
    template <typename T> using mat3x3 = mat3x3_t<ASX::ID_value, T>;
    template <typename T> using mat2x2 = mat2x2_t<ASX::ID_value, T>;

    typedef mat4x4<double>   mat4x4d;
    typedef mat4x4<float>    mat4x4f;
    typedef mat4x4<int32_t>	 mat4x4i;
    typedef mat4x4<uint32_t> mat4x4u;

    typedef mat3x3<double>   mat3x3d;
    typedef mat3x3<float>    mat3x3f;
    typedef mat3x3<int32_t>	 mat3x3i;
    typedef mat3x3<uint32_t> mat3x3u;

    typedef mat2x2<double>   mat2x2d;
    typedef mat2x2<float>    mat2x2f;
    typedef mat2x2<int32_t>	 mat2x2i;
    typedef mat2x2<uint32_t> mat2x2u;

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, typename T>
    struct CML_ALIGN(sizeof(T)*16) mat4x4_t
    {
        static_assert(sizeof(T) == 4, "T is not 4 bytes");

        typedef ASX::ASAGroup<T, t_id> ASX_ASA;

        union { T t00; ASX_ASA dummy1; };
        union { T t01; ASX_ASA dummy2; };
        union { T t02; ASX_ASA dummy3; };
        union { T t03; ASX_ASA dummy4; };

        union { T t10; ASX_ASA dummy5; };
        union { T t11; ASX_ASA dummy6; };
        union { T t12; ASX_ASA dummy7; };
        union { T t13; ASX_ASA dummy8; };

        union { T t20; ASX_ASA dummy9; };
        union { T t21; ASX_ASA dummy10; };
        union { T t22; ASX_ASA dummy11; };
        union { T t23; ASX_ASA dummy12; };

        union { T t30; ASX_ASA dummy13; };
        union { T t31; ASX_ASA dummy14; };
        union { T t32; ASX_ASA dummy15; };
        union { T t33; ASX_ASA dummy16; };

        /* ------------------------------ Constructors ------------------------------ */

        CML_FUNC_DECL CML_CONSTEXPR mat4x4_t() :
            t00(0), t01(0), t02(0), t03(0),
            t10(0), t11(0), t12(0), t13(0),
            t20(0), t21(0), t22(0), t23(0),
            t30(0), t31(0), t32(0), t33(0) {};

        CML_FUNC_DECL CML_CONSTEXPR mat4x4_t(T x) :
            t00(x), t01(x), t02(x), t03(x),
            t10(x), t11(x), t12(x), t13(x),
            t20(x), t21(x), t22(x), t23(x),
            t30(x), t31(x), t32(x), t33(x) {};

        CML_FUNC_DECL CML_CONSTEXPR mat4x4_t(
            T t00, T t01, T t02, T t03,
            T t10, T t11, T t12, T t13,
            T t20, T t21, T t22, T t23,
            T t30, T t31, T t32, T t33) :
            t00(t00), t01(t01), t02(t02), t03(t03),
            t10(t10), t11(t11), t12(t12), t13(t13),
            t20(t20), t21(t21), t22(t22), t23(t23),
            t30(t30), t31(t31), t32(t32), t33(t33) {};

        template<ASX::ID u_id, class U>
        CML_FUNC_DECL CML_CONSTEXPR mat4x4_t(const mat4x4_t<u_id, U>&m)
        {
            t00 = m.t00; t01 = m.t01; t02 = m.t02; t03 = m.t03;
            t10 = m.t10; t11 = m.t11; t12 = m.t12; t13 = m.t13;
            t20 = m.t20; t21 = m.t21; t22 = m.t22; t23 = m.t23;
            t30 = m.t30; t31 = m.t31; t32 = m.t32; t33 = m.t33;
        };

        template<ASX::ID u_id, ASX::ID v_id, class U, class V>
        CML_FUNC_DECL CML_CONSTEXPR mat4x4_t(
            const vec4_t<u_id, U>&v0,
            const vec4_t<v_id, V>&v1,
            const vec4_t<v_id, V>&v2,
            const vec4_t<v_id, V>&v3)
        {
            t00 = v0.x; t01 = v0.y; t02 = v0.z; t03 = v0.w;
            t10 = v1.x; t11 = v1.y; t12 = v1.z; t13 = v1.w;
            t20 = v2.x; t21 = v2.y; t22 = v2.z; t23 = v2.w;
            t30 = v3.x; t31 = v3.y; t32 = v3.z; t33 = v3.w;
        };

        CML_FUNC_DECL CML_CONSTEXPR mat4x4_t(const glm::mat4&m)
        {
            t00 = m[0][0]; t01 = m[0][1]; t02 = m[0][2]; t03 = m[0][3];
            t10 = m[1][0]; t11 = m[1][1]; t12 = m[1][2]; t13 = m[1][3];
            t20 = m[2][0]; t21 = m[2][1]; t22 = m[2][2]; t23 = m[2][3];
            t30 = m[3][0]; t31 = m[3][1]; t32 = m[3][2]; t33 = m[3][3];
        };

        /* ------------------------------- Assignment ------------------------------- */

        template<ASX::ID u_id, class U>
        CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T>& operator=(const mat4x4_t<u_id, U>&other)
        {
            t00 = static_cast<T>(other.t00); t01 = static_cast<T>(other.t01); t02 = static_cast<T>(other.t02); t03 = static_cast<T>(other.t03);
            t10 = static_cast<T>(other.t10); t11 = static_cast<T>(other.t11); t12 = static_cast<T>(other.t12); t13 = static_cast<T>(other.t13);
            t20 = static_cast<T>(other.t20); t21 = static_cast<T>(other.t21); t22 = static_cast<T>(other.t22); t23 = static_cast<T>(other.t23);
            t30 = static_cast<T>(other.t30); t31 = static_cast<T>(other.t31); t32 = static_cast<T>(other.t32); t33 = static_cast<T>(other.t33);
            return *this;
        };

        /* --------------------------------- Casting -------------------------------- */

        // TODO

        /* ---------------------------------- Util ---------------------------------- */

        CML_FUNC_DECL const T& operator()(const uint32_t row, const uint32_t col) const
        {
            uint32_t index = row*4+col;

            return
                (static_cast<T>(index==0)*t00)+
                (static_cast<T>(index==1)*t01)+
                (static_cast<T>(index==2)*t02)+
                (static_cast<T>(index==3)*t03)+

                (static_cast<T>(index==4)*t10)+
                (static_cast<T>(index==5)*t11)+
                (static_cast<T>(index==6)*t12)+
                (static_cast<T>(index==7)*t13)+

                (static_cast<T>(index==8)*t20)+
                (static_cast<T>(index==9)*t21)+
                (static_cast<T>(index==10)*t22)+
                (static_cast<T>(index==11)*t23)+

                (static_cast<T>(index==12)*t30)+
                (static_cast<T>(index==13)*t31)+
                (static_cast<T>(index==14)*t32)+
                (static_cast<T>(index==15)*t33);
        };

        CML_FUNC_DECL CML_CONSTEXPR void print() const
        {
            printf("[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n\n",
                static_cast<float>(t00), static_cast<float>(t01), static_cast<float>(t02), static_cast<float>(t03),
                static_cast<float>(t10), static_cast<float>(t11), static_cast<float>(t12), static_cast<float>(t13),
                static_cast<float>(t20), static_cast<float>(t21), static_cast<float>(t22), static_cast<float>(t23),
                static_cast<float>(t30), static_cast<float>(t31), static_cast<float>(t32), static_cast<float>(t33)
            );
        };

        static CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> zero(void)
        {
            return mat4x4<T>(0);
        }

        static CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> ones(void)
        {
            return mat4x4<T>(1);
        }

        static CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> unit(void)
        {
            return mat4x4<T>(
                1, 0, 0, 0,
                0, 1, 0, 0,
                0, 0, 1, 0,
                0, 0, 0, 1);
        }
    };

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, typename T>
    struct CML_ALIGN(sizeof(T)*16) mat3x3_t
    {
        static_assert(sizeof(T) == 4, "T is not 4 bytes");

        typedef ASX::ASAGroup<T, t_id> ASX_ASA;

        union { T t00; ASX_ASA dummy1; };
        union { T t01; ASX_ASA dummy2; };
        union { T t02; ASX_ASA dummy3; };

        union { T t10; ASX_ASA dummy4; };
        union { T t11; ASX_ASA dummy5; };
        union { T t12; ASX_ASA dummy6; };

        union { T t20; ASX_ASA dummy7; };
        union { T t21; ASX_ASA dummy8; };
        union { T t22; ASX_ASA dummy9; };

        /* ------------------------------ Constructors ------------------------------ */

        CML_FUNC_DECL CML_CONSTEXPR mat3x3_t() :
            t00(0), t01(0), t02(0),
            t10(0), t11(0), t12(0),
            t20(0), t21(0), t22(0) {};

        CML_FUNC_DECL CML_CONSTEXPR mat3x3_t(T x) :
            t00(x), t01(x), t02(x),
            t10(x), t11(x), t12(x),
            t20(x), t21(x), t22(x) {};

        CML_FUNC_DECL CML_CONSTEXPR mat3x3_t(
            T t00, T t01, T t02,
            T t10, T t11, T t12,
            T t20, T t21, T t22) :
            t00(t00), t01(t01), t02(t02),
            t10(t10), t11(t11), t12(t12),
            t20(t20), t21(t21), t22(t22) {};

        template<ASX::ID u_id, class U>
        CML_FUNC_DECL CML_CONSTEXPR mat3x3_t(const mat3x3_t<u_id, U>&m)
        {
            t00 = m.t00; t01 = m.t01; t02 = m.t02;
            t10 = m.t10; t11 = m.t11; t12 = m.t12;
            t20 = m.t20; t21 = m.t21; t22 = m.t22;
        };

        template<ASX::ID u_id, ASX::ID v_id, class U, class V>
        CML_FUNC_DECL CML_CONSTEXPR mat3x3_t(
            const vec3_t<u_id, U>&v0,
            const vec3_t<v_id, V>&v1,
            const vec3_t<v_id, V>&v2)
        {
            t00 = v0.x; t01 = v0.y; t02 = v0.z;
            t10 = v1.x; t11 = v1.y; t12 = v1.z;
            t20 = v2.x; t21 = v2.y; t22 = v2.z;
        };

        CML_FUNC_DECL CML_CONSTEXPR mat3x3_t(const glm::mat2&m)
        {
            t00 = m[0][0]; t01 = m[0][1]; t02 = m[0][2];
            t10 = m[1][0]; t11 = m[1][1]; t12 = m[1][2];
            t20 = m[2][0]; t21 = m[2][1]; t22 = m[2][2];
        };

        /* ------------------------------- Assignment ------------------------------- */

        template<ASX::ID u_id, class U>
        CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T>& operator=(const mat3x3_t<u_id, U>&other)
        {
            t00 = static_cast<T>(other.t00); t01 = static_cast<T>(other.t01); t02 = static_cast<T>(other.t02);
            t10 = static_cast<T>(other.t10); t11 = static_cast<T>(other.t11); t12 = static_cast<T>(other.t12);
            t20 = static_cast<T>(other.t20); t21 = static_cast<T>(other.t21); t22 = static_cast<T>(other.t22);
            return *this;
        };

        /* --------------------------------- Casting -------------------------------- */

        // TODO

        /* ---------------------------------- Util ---------------------------------- */

        CML_FUNC_DECL const T& operator()(const uint32_t row, const uint32_t col) const
        {
            uint32_t index = row*3+col;

            return
                (static_cast<T>(index==0)*t00)+
                (static_cast<T>(index==1)*t01)+
                (static_cast<T>(index==2)*t02)+
                (static_cast<T>(index==3)*t10)+
                (static_cast<T>(index==4)*t11)+
                (static_cast<T>(index==5)*t12)+
                (static_cast<T>(index==6)*t20)+
                (static_cast<T>(index==7)*t21)+
                (static_cast<T>(index==8)*t22);
        };

        CML_FUNC_DECL CML_CONSTEXPR void print() const
        {
            printf("[%f,%f,%f]\n[%f,%f,%f]\n[%f,%f,%f]\n\n",
                static_cast<float>(t00), static_cast<float>(t01), static_cast<float>(t02),
                static_cast<float>(t10), static_cast<float>(t11), static_cast<float>(t12),
                static_cast<float>(t20), static_cast<float>(t21), static_cast<float>(t22)
            );
        };

        static CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> zero(void)
        {
            return mat3x3<T>(0);
        }

        static CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> ones(void)
        {
            return mat3x3<T>(1);
        }

        static CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> unit(void)
        {
            return mat3x3<T>(
                1, 0, 0,
                0, 1, 0,
                0, 0, 1);
        }
    };

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, typename T>
    struct CML_ALIGN(sizeof(T)*4) mat2x2_t
    {
        static_assert(sizeof(T)==4, "T is not 4 bytes");

        typedef ASX::ASAGroup<T, t_id> ASX_ASA;

        union { T t00; ASX_ASA dummy1; };
        union { T t01; ASX_ASA dummy2; };
        union { T t10; ASX_ASA dummy3; };
        union { T t11; ASX_ASA dummy4; };

        /* ------------------------------ Constructors ------------------------------ */

        CML_FUNC_DECL CML_CONSTEXPR mat2x2_t() :
            t00(0), t01(0),
            t10(0), t11(0) {};

        CML_FUNC_DECL CML_CONSTEXPR mat2x2_t(T x) :
            t00(x), t01(x),
            t10(x), t11(x) {};

        CML_FUNC_DECL CML_CONSTEXPR mat2x2_t(T t00, T t01, T t10, T t11) :
            t00(t00), t01(t01),
            t10(t10), t11(t11) {};

        template<ASX::ID u_id, class U>
        CML_FUNC_DECL CML_CONSTEXPR mat2x2_t(const mat2x2_t<u_id, U>&m)
        {
            t00 = m.t00; t01 = m.t01;
            t10 = m.t10; t11 = m.t11;
        };

        template<ASX::ID u_id, ASX::ID v_id, class U, class V>
        CML_FUNC_DECL CML_CONSTEXPR mat2x2_t(const vec2_t<u_id, U>&v0, const vec2_t<v_id, V>&v1)
        {
            t00 = v0.x; t01 = v0.y;
            t10 = v1.x; t11 = v1.y;
        };

        CML_FUNC_DECL CML_CONSTEXPR mat2x2_t(const glm::mat2&m)
        {
            t00 = m[0][0]; t01 = m[0][1];
            t10 = m[1][0]; t11 = m[1][0];
        };

        /* ------------------------------- Assignment ------------------------------- */

        template<ASX::ID u_id, class U>
        CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T>& operator=(const mat2x2_t<u_id, U>&other)
        {
            t00 = static_cast<T>(other.t00);
            t01 = static_cast<T>(other.t01);
            t10 = static_cast<T>(other.t10);
            t11 = static_cast<T>(other.t11);
            return *this;
        };

        /* --------------------------------- Casting -------------------------------- */

        // TODO

        /* ---------------------------------- Util ---------------------------------- */

        CML_FUNC_DECL const T& operator()(const uint32_t row, const uint32_t col) const
        {
            uint32_t index = row*2+col;
            return
                (static_cast<T>(index==0)*t00)+
                (static_cast<T>(index==1)*t01)+
                (static_cast<T>(index==2)*t10)+
                (static_cast<T>(index==3)*t11);
        };

        CML_FUNC_DECL CML_CONSTEXPR void print() const
        {
            printf("[%f,%f]\n[%f,%f]\n\n",
                static_cast<float>(t00), static_cast<float>(t01),
                static_cast<float>(t10), static_cast<float>(t11)
            );
        };

        static CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> zero(void)
        {
            return mat2x2<T>(0);
        }

        static CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> ones(void)
        {
            return mat2x2<T>(1);
        }

        static CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> unit(void)
        {
            return mat2x2<T>(
                1, 0,
                0, 1);
        }
    };

    /* -------------------------------------------------------------------------- */
    /*                                 Comparators                                */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator==(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, U>& m2);

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, U>& m2);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator==(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, U>& m2);

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, U>& m2);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator==(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, U>& m2);

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, U>& m2);

    /* -------------------------------------------------------------------------- */
    /*                               Multiplication                               */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator*=(mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator*=(mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T> operator*=(vec4_t<t_id, T>& v, const mat4x4_t<u_id, T>& m);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator*(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator*(const vec4_t<t_id, T>& v, const mat4x4_t<u_id, T>& m);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator*(const mat4x4_t<t_id, T>& m, const vec4_t<u_id, T>& v);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator*(const mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator*(const T s, const mat4x4_t<t_id, T>& m);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator*=(mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator*=(mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T> operator*=(vec3_t<t_id, T>& v, const mat3x3_t<u_id, T>& m);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator*(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator*(const vec3_t<t_id, T>& v, const mat3x3_t<u_id, T>& m);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator*(const mat3x3_t<t_id, T>& m, const vec3_t<u_id, T>& v);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator*(const mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator*(const T s, const mat3x3_t<t_id, T>& m);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator*=(mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator*=(mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T> operator*=(vec2_t<t_id, T>& v, const mat2x2_t<u_id, T>& m);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator*(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator*(const vec2_t<t_id, T>& v, const mat2x2_t<u_id, T>& m);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator*(const mat2x2_t<t_id, T>& m, const vec2_t<u_id, T>& v);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator*(const mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator*(const T s, const mat2x2_t<t_id, T>& m);

    /* -------------------------------------------------------------------------- */
    /*                                  Division                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator/=(mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator/=(mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator/(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator/(const mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator/(const T s, const mat4x4_t<t_id, T>& m);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator/=(mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator/=(mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator/(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator/(const mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator/(const T s, const mat3x3_t<t_id, T>& m);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator/=(mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator/=(mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator/(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator/(const mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator/(const T s, const mat2x2_t<t_id, T>& m);

    /* -------------------------------------------------------------------------- */
    /*                                  Addition                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator+=(mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator+=(mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator+(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator+(const mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator+(const T s, const mat4x4_t<t_id, T>& m);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator+=(mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator+=(mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator+(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator+(const mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator+(const T s, const mat3x3_t<t_id, T>& m);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator+=(mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator+=(mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator+(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator+(const mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator+(const T s, const mat2x2_t<t_id, T>& m);

    /* -------------------------------------------------------------------------- */
    /*                                 Subtraction                                */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator-=(mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator-=(mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator-(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator-(const mat4x4_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator-(const T s, const mat4x4_t<t_id, T>& m);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator-=(mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator-=(mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator-(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator-(const mat3x3_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator-(const T s, const mat3x3_t<t_id, T>& m);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator-=(mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator-=(mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator-(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator-(const mat2x2_t<t_id, T>& m, const T s);

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator-(const T s, const mat2x2_t<t_id, T>& m);

    /* -------------------------------------------------------------------------- */
    /*                                  Negation                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator-(const mat4x4_t<t_id, T>& m1);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator-(const mat3x3_t<t_id, T>& m1);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator-(const mat2x2_t<t_id, T>& m1);

    /* -------------------------------------------------------------------------- */
    /*                                  Transpose                                 */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> transpose(const mat4x4_t<t_id, T>& m);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> transpose(const mat3x3_t<t_id, T>& m);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> transpose(const mat2x2_t<t_id, T>& m);

    /* -------------------------------------------------------------------------- */
    /*                                 Determinant                                */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR T det(const mat4x4_t<t_id, T>& m);
    
    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR T det(const mat3x3_t<t_id, T>& m);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR T det(const mat2x2_t<t_id, T>& m);

    /* -------------------------------------------------------------------------- */
    /*                                   Inverse                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> inv(const mat4x4_t<t_id, T>& m);

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> inv(const mat3x3_t<t_id, T>& m);

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> inv(const mat2x2_t<t_id, T>& m);

} // namespace cml

#include "mat.inl"

#endif // _CML_MAT_