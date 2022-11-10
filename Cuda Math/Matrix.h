#ifndef _JEK_MATRIX_
#define _JEK_MATRIX_

#include "Vector.h"

namespace jek
{
    template <class T> struct Matrix4x4;
    template <class T> struct Matrix3x3;
    template <class T> struct Matrix2x2;

    template <class T> struct _ALIGN(64) Matrix4x4
    {
        static_assert(sizeof(T) == 4, "T is not 4 bytes");

        T m[4][4];
        /*
        T t00;
        T t01;
        T t02; 
        T t03;
        T t10; 
        T t11; 
        T t12; 
        T t13;
        T t20;
        T t21; 
        T t22; 
        T t23;
        T t30; 
        T t31; 
        T t32; 
        T t33;
        */

        _HOST_DEVICE Matrix4x4()
        {
            m[0][0] = m[1][1] = m[2][2] = m[3][3] = 1;
            m[0][1] = m[0][2] = m[0][3] = m[1][0] =
            m[1][2] = m[1][3] = m[2][0] = m[2][1] =
            m[2][3] = m[3][0] = m[3][1] = m[3][2] = 0;
        };
        _HOST_DEVICE Matrix4x4(T x)
        {
            m[0][0] = m[1][1] = m[2][2] = m[3][3] =
            m[0][1] = m[0][2] = m[0][3] = m[1][0] =
            m[1][2] = m[1][3] = m[2][0] = m[2][1] =
            m[2][3] = m[3][0] = m[3][1] = m[3][2] = x;
        };
        _HOST_DEVICE Matrix4x4(
            T t00, T t01, T t02, T t03,
            T t10, T t11, T t12, T t13,
            T t20, T t21, T t22, T t23,
            T t30, T t31, T t32, T t33)
        {
            m[0][0] = t00; m[0][1] = t01; m[0][2] = t02; m[0][3] = t03;
            m[1][0] = t10; m[1][1] = t11; m[1][2] = t12; m[1][3] = t13;
            m[2][0] = t20; m[2][1] = t21; m[2][2] = t22; m[2][3] = t23;
            m[3][0] = t30; m[3][1] = t31; m[3][2] = t32; m[3][3] = t33;
        };
        _HOST_DEVICE Matrix4x4(
            const Vec4<T>&v0,
            const Vec4<T>&v1,
            const Vec4<T>&v2,
            const Vec4<T>&v3)
        {
            m[0][0] = v0.x; m[0][1] = v0.y; m[0][2] = v0.z; m[0][3] = v0.w;
            m[1][0] = v1.x; m[1][1] = v1.y; m[1][2] = v1.z; m[1][3] = v1.w;
            m[2][0] = v2.x; m[2][1] = v2.y; m[2][2] = v2.z; m[2][3] = v2.w;
            m[3][0] = v3.x; m[3][1] = v3.y; m[3][2] = v3.z; m[3][3] = v3.w;
        };
        _HOST Matrix4x4(const glm::mat4 & _m)
        {
            m[0][0] = _m[0][0]; m[0][1] = _m[0][1]; m[0][2] = _m[0][2]; m[0][3] = _m[0][3];
            m[1][0] = _m[1][0]; m[1][1] = _m[1][1]; m[1][2] = _m[1][2]; m[1][3] = _m[1][3];
            m[2][0] = _m[2][0]; m[2][1] = _m[2][1]; m[2][2] = _m[2][2]; m[2][3] = _m[2][3];
            m[3][0] = _m[3][0]; m[3][1] = _m[3][1]; m[3][2] = _m[3][2]; m[3][3] = _m[3][3];
        };

        operator Matrix3x3<T>() const
        {

        };

        _HOST_DEVICE void print()
        {
            printf("[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n[%f,%f,%f,%f]\n\n",
                m[0][0], m[0][1], m[0][2], m[0][3],
                m[1][0], m[1][1], m[1][2], m[1][3],
                m[2][0], m[2][1], m[2][2], m[2][3],
                m[3][0], m[3][1], m[3][2], m[3][3]
            );
        };
    };

    template <class T> struct _ALIGN(64) Matrix3x3
    {
        static_assert(sizeof(T) == 4, "T is not 4 bytes");

        T m[3][3];

        _HOST_DEVICE Matrix3x3();
        _HOST_DEVICE Matrix3x3(T x);
        _HOST_DEVICE Matrix3x3(
            T t00, T t01, T t02,
            T t10, T t11, T t12,
            T t20, T t21, T t22);
        _HOST_DEVICE Matrix3x3(
            const Vec3<T>&v0,
            const Vec3<T>&v1,
            const Vec3<T>&v2);
        _HOST Matrix3x3(const glm::mat3 & m);

        operator Matrix4x4<T>() const;

        _HOST_DEVICE void print();
    };
    template <class T> struct _ALIGN(16) Matrix2x2
    {
        static_assert(sizeof(T) == 4, "T is not 4 bytes");

        T m[2][2];

        _HOST_DEVICE Matrix2x2();
        _HOST_DEVICE Matrix2x2(T x);
        _HOST_DEVICE Matrix2x2(
            T t00, T t01,
            T t10, T t11);
        _HOST_DEVICE Matrix2x2(
            const Vec2<T>&v0,
            const Vec2<T>&v1);
        _HOST Matrix2x2(const glm::mat2 & m);

        _HOST_DEVICE void print();
    };

    typedef Matrix4x4<float>    Matrix4x4f;
    typedef Matrix4x4<int32_t>	Matrix4x4i;
    typedef Matrix4x4<uint32_t>	Matrix4x4u;

    typedef Matrix3x3<float>    Matrix3x3f;
    typedef Matrix3x3<int32_t>	Matrix3x3i;
    typedef Matrix3x3<uint32_t>	Matrix3x3u;

    typedef Matrix2x2<float>    Matrix2x2f;
    typedef Matrix2x2<int32_t>	Matrix2x2i;
    typedef Matrix2x2<uint32_t>	Matrix2x2u;

    /* -------------------------------------------------------------------------- */
    /*                                 Comparators                                */
    /* -------------------------------------------------------------------------- */

    template<class T> _HOST_DEVICE bool operator==(const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (m1.m[i][j] != m2.m[i][j]) return false;
        return true;
    }
    template<class T> _HOST_DEVICE bool operator!=(const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                if (m1.m[i][j] != m2.m[i][j]) return true;
        return false;
    }

    /* -------------------------------------------------------------------------- */
    /*                               Multiplication                               */
    /* -------------------------------------------------------------------------- */

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> operator*(const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
        Matrix4x4<T> out;
        for (int i = 0; i < 4; i++) {
            const float ai0 = m1.m[i][0], ai1 = m1.m[i][1], ai2 = m1.m[i][2], ai3 = m1.m[i][3];
            out.m[i][0] = ai0 * m2.m[0][0] + ai1 * m2.m[1][0] + ai2 * m2.m[2][0] + ai3 * m2.m[3][0];
            out.m[i][1] = ai0 * m2.m[0][1] + ai1 * m2.m[1][1] + ai2 * m2.m[2][1] + ai3 * m2.m[3][1];
            out.m[i][2] = ai0 * m2.m[0][2] + ai1 * m2.m[1][2] + ai2 * m2.m[2][2] + ai3 * m2.m[3][2];
            out.m[i][3] = ai0 * m2.m[0][3] + ai1 * m2.m[1][3] + ai2 * m2.m[2][3] + ai3 * m2.m[3][3];
        }
        return out;
    }
    template<class T> _HOST_DEVICE
    inline Vec4<T> operator*(const Vec4<T>& v, const Matrix4x4<T>& m)
    {
        Vec4<T> out;
        out.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3] * v.w;
        out.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3] * v.w;
        out.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3] * v.w;
        out.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3] * v.w;
        return out;
    }
    template<class T> _HOST_DEVICE
    inline Vec4<T> operator*(const Matrix4x4<T>& m, const Vec4<T>& v)
    {
        Vec4<T> out;
        out.x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0] * v.w;
        out.y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1] * v.w;
        out.z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2] * v.w;
        out.w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3] * v.w;
        return out;
    }
    template<class T> _HOST_DEVICE
    inline Vec3<T> operator*(const Vec3<T>& v, const Matrix4x4<T>& m)
    {
        Vec4<T> out;
        out.x = m.m[0][0] * v.x + m.m[0][1] * v.y + m.m[0][2] * v.z + m.m[0][3];
        out.y = m.m[1][0] * v.x + m.m[1][1] * v.y + m.m[1][2] * v.z + m.m[1][3];
        out.z = m.m[2][0] * v.x + m.m[2][1] * v.y + m.m[2][2] * v.z + m.m[2][3];
        out.w = m.m[3][0] * v.x + m.m[3][1] * v.y + m.m[3][2] * v.z + m.m[3][3];

        return (out.w == 0.0) ? Vec3f(0) : 1.f / out.w * Vec3<T>(out.x, out.y, out.z);
    }
    template<class T> _HOST_DEVICE
    inline Vec3<T> operator*(const Matrix4x4<T>& m, const Vec3<T>& v)
    {
        Vec4<T> out;
        out.x = m.m[0][0] * v.x + m.m[1][0] * v.y + m.m[2][0] * v.z + m.m[3][0];
        out.y = m.m[0][1] * v.x + m.m[1][1] * v.y + m.m[2][1] * v.z + m.m[3][1];
        out.z = m.m[0][2] * v.x + m.m[1][2] * v.y + m.m[2][2] * v.z + m.m[3][2];
        out.w = m.m[0][3] * v.x + m.m[1][3] * v.y + m.m[2][3] * v.z + m.m[3][3];

        return (out.w == 0.0) ? Vec3f(0) : 1.f / out.w * Vec3<T>(out.x, out.y, out.z);
    }

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> operator*(const T s, const Matrix4x4<T>& m)
    {
        Matrix4x4<T> out;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out.m[i][j] = s * m.m[i][j];
        return out;
    }
    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> operator*(const Matrix4x4<T>& m, const T s)
    {
        Matrix4x4<T> out;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out.m[i][j] = s * m.m[i][j];
        return out;
    }

    /* -------------------------------------------------------------------------- */
    /*                                  Division                                  */
    /* -------------------------------------------------------------------------- */

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> operator/(const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
        Matrix4x4<T> out;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out.m[i][j] = m1.m[i][j] / m2.m[i][j];
        return out;
    }

    /* -------------------------------------------------------------------------- */
    /*                                  Addition                                  */
    /* -------------------------------------------------------------------------- */

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> operator+(const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
        Matrix4x4<T> out;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out.m[i][j] = m1.m[i][j] + m2.m[i][j];
        return out;
    }

    /* -------------------------------------------------------------------------- */
    /*                                 Subtraction                                */
    /* -------------------------------------------------------------------------- */

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> operator-(const Matrix4x4<T>& m1, const Matrix4x4<T>& m2)
    {
        Matrix4x4<T> out;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out.m[i][j] = m1.m[i][j] - m2.m[i][j];
        return out;
    }

    /* -------------------------------------------------------------------------- */
    /*                                  Negation                                  */
    /* -------------------------------------------------------------------------- */

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> operator-(const Matrix4x4<T>& m1)
    {
        Matrix4x4<T> out;
        for (int i = 0; i < 4; ++i)
            for (int j = 0; j < 4; ++j)
                out.m[i][j] = -m1.m[i][j];
        return out;
    }

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> transpose(const Matrix4x4<T>& m)
    {
        return Matrix4x4<T>(
            m.m[0][0], m.m[1][0], m.m[2][0], m.m[3][0],
            m.m[0][1], m.m[1][1], m.m[2][1], m.m[3][1],
            m.m[0][2], m.m[1][2], m.m[2][2], m.m[3][2],
            m.m[0][3], m.m[1][3], m.m[2][3], m.m[3][3]);
    }
    template<class T> _HOST_DEVICE
    inline T det(const Matrix4x4<T>& m)
    {
        T d =
            m.m[3][0] * m.m[2][1] * m.m[1][2] * m.m[0][3] - m.m[2][0] * m.m[3][1] * m.m[1][2] * m.m[0][3] -
            m.m[3][0] * m.m[1][1] * m.m[2][2] * m.m[0][3] + m.m[1][0] * m.m[3][1] * m.m[2][2] * m.m[0][3] +
            m.m[2][0] * m.m[1][1] * m.m[3][2] * m.m[0][3] - m.m[1][0] * m.m[2][1] * m.m[3][2] * m.m[0][3] -
            m.m[3][0] * m.m[2][1] * m.m[0][2] * m.m[1][3] + m.m[2][0] * m.m[3][1] * m.m[0][2] * m.m[1][3] +
            m.m[3][0] * m.m[0][1] * m.m[2][2] * m.m[1][3] - m.m[0][0] * m.m[3][1] * m.m[2][2] * m.m[1][3] -
            m.m[2][0] * m.m[0][1] * m.m[3][2] * m.m[1][3] + m.m[0][0] * m.m[2][1] * m.m[3][2] * m.m[1][3] +
            m.m[3][0] * m.m[1][1] * m.m[0][2] * m.m[2][3] - m.m[1][0] * m.m[3][1] * m.m[0][2] * m.m[2][3] -
            m.m[3][0] * m.m[0][1] * m.m[1][2] * m.m[2][3] + m.m[0][0] * m.m[3][1] * m.m[1][2] * m.m[2][3] +
            m.m[1][0] * m.m[0][1] * m.m[3][2] * m.m[2][3] - m.m[0][0] * m.m[1][1] * m.m[3][2] * m.m[2][3] -
            m.m[2][0] * m.m[1][1] * m.m[0][2] * m.m[3][3] + m.m[1][0] * m.m[2][1] * m.m[0][2] * m.m[3][3] +
            m.m[2][0] * m.m[0][1] * m.m[1][2] * m.m[3][3] - m.m[0][0] * m.m[2][1] * m.m[1][2] * m.m[3][3] -
            m.m[1][0] * m.m[0][1] * m.m[2][2] * m.m[3][3] + m.m[0][0] * m.m[1][1] * m.m[2][2] * m.m[3][3];

        return d;
    }
    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> inv(const Matrix4x4<T>& m)
    {
        Matrix4x4<T> inv;
        double det;

        inv.m[0][0] = m.m[1][1] * m.m[2][2] * m.m[3][3] -
            m.m[1][1] * m.m[2][3] * m.m[3][2] -
            m.m[2][1] * m.m[1][2] * m.m[3][3] +
            m.m[2][1] * m.m[1][3] * m.m[3][2] +
            m.m[3][1] * m.m[1][2] * m.m[2][3] -
            m.m[3][1] * m.m[1][3] * m.m[2][2];

        inv.m[1][0] = -m.m[1][0] * m.m[2][2] * m.m[3][3] +
            m.m[1][0] * m.m[2][3] * m.m[3][2] +
            m.m[2][0] * m.m[1][2] * m.m[3][3] -
            m.m[2][0] * m.m[1][3] * m.m[3][2] -
            m.m[3][0] * m.m[1][2] * m.m[2][3] +
            m.m[3][0] * m.m[1][3] * m.m[2][2];

        inv.m[2][0] = m.m[1][0] * m.m[2][1] * m.m[3][3] -
            m.m[1][0] * m.m[2][3] * m.m[3][1] -
            m.m[2][0] * m.m[1][1] * m.m[3][3] +
            m.m[2][0] * m.m[1][3] * m.m[3][1] +
            m.m[3][0] * m.m[1][1] * m.m[2][3] -
            m.m[3][0] * m.m[1][3] * m.m[2][1];

        inv.m[3][0] = -m.m[1][0] * m.m[2][1] * m.m[3][2] +
            m.m[1][0] * m.m[2][2] * m.m[3][1] +
            m.m[2][0] * m.m[1][1] * m.m[3][2] -
            m.m[2][0] * m.m[1][2] * m.m[3][1] -
            m.m[3][0] * m.m[1][1] * m.m[2][2] +
            m.m[3][0] * m.m[1][2] * m.m[2][1];

        inv.m[0][1] = -m.m[0][1] * m.m[2][2] * m.m[3][3] +
            m.m[0][1] * m.m[2][3] * m.m[3][2] +
            m.m[2][1] * m.m[0][2] * m.m[3][3] -
            m.m[2][1] * m.m[0][3] * m.m[3][2] -
            m.m[3][1] * m.m[0][2] * m.m[2][3] +
            m.m[3][1] * m.m[0][3] * m.m[2][2];

        inv.m[1][1] = m.m[0][0] * m.m[2][2] * m.m[3][3] -
            m.m[0][0] * m.m[2][3] * m.m[3][2] -
            m.m[2][0] * m.m[0][2] * m.m[3][3] +
            m.m[2][0] * m.m[0][3] * m.m[3][2] +
            m.m[3][0] * m.m[0][2] * m.m[2][3] -
            m.m[3][0] * m.m[0][3] * m.m[2][2];

        inv.m[2][1] = -m.m[0][0] * m.m[2][1] * m.m[3][3] +
            m.m[0][0] * m.m[2][3] * m.m[3][1] +
            m.m[2][0] * m.m[0][1] * m.m[3][3] -
            m.m[2][0] * m.m[0][3] * m.m[3][1] -
            m.m[3][0] * m.m[0][1] * m.m[2][3] +
            m.m[3][0] * m.m[0][3] * m.m[2][1];

        inv.m[3][1] = m.m[0][0] * m.m[2][1] * m.m[3][2] -
            m.m[0][0] * m.m[2][2] * m.m[3][1] -
            m.m[2][0] * m.m[0][1] * m.m[3][2] +
            m.m[2][0] * m.m[0][2] * m.m[3][1] +
            m.m[3][0] * m.m[0][1] * m.m[2][2] -
            m.m[3][0] * m.m[0][2] * m.m[2][1];

        inv.m[0][2] = m.m[0][1] * m.m[1][2] * m.m[3][3] -
            m.m[0][1] * m.m[1][3] * m.m[3][2] -
            m.m[1][1] * m.m[0][2] * m.m[3][3] +
            m.m[1][1] * m.m[0][3] * m.m[3][2] +
            m.m[3][1] * m.m[0][2] * m.m[1][3] -
            m.m[3][1] * m.m[0][3] * m.m[1][2];

        inv.m[1][2] = -m.m[0][0] * m.m[1][2] * m.m[3][3] +
            m.m[0][0] * m.m[1][3] * m.m[3][2] +
            m.m[1][0] * m.m[0][2] * m.m[3][3] -
            m.m[1][0] * m.m[0][3] * m.m[3][2] -
            m.m[3][0] * m.m[0][2] * m.m[1][3] +
            m.m[3][0] * m.m[0][3] * m.m[1][2];

        inv.m[2][2] = m.m[0][0] * m.m[1][1] * m.m[3][3] -
            m.m[0][0] * m.m[1][3] * m.m[3][1] -
            m.m[1][0] * m.m[0][1] * m.m[3][3] +
            m.m[1][0] * m.m[0][3] * m.m[3][1] +
            m.m[3][0] * m.m[0][1] * m.m[1][3] -
            m.m[3][0] * m.m[0][3] * m.m[1][1];

        inv.m[3][2] = -m.m[0][0] * m.m[1][1] * m.m[3][2] +
            m.m[0][0] * m.m[1][2] * m.m[3][1] +
            m.m[1][0] * m.m[0][1] * m.m[3][2] -
            m.m[1][0] * m.m[0][2] * m.m[3][1] -
            m.m[3][0] * m.m[0][1] * m.m[1][2] +
            m.m[3][0] * m.m[0][2] * m.m[1][1];

        inv.m[0][3] = -m.m[0][1] * m.m[1][2] * m.m[2][3] +
            m.m[0][1] * m.m[1][3] * m.m[2][2] +
            m.m[1][1] * m.m[0][2] * m.m[2][3] -
            m.m[1][1] * m.m[0][3] * m.m[2][2] -
            m.m[2][1] * m.m[0][2] * m.m[1][3] +
            m.m[2][1] * m.m[0][3] * m.m[1][2];

        inv.m[1][3] = m.m[0][0] * m.m[1][2] * m.m[2][3] -
            m.m[0][0] * m.m[1][3] * m.m[2][2] -
            m.m[1][0] * m.m[0][2] * m.m[2][3] +
            m.m[1][0] * m.m[0][3] * m.m[2][2] +
            m.m[2][0] * m.m[0][2] * m.m[1][3] -
            m.m[2][0] * m.m[0][3] * m.m[1][2];

        inv.m[2][3] = -m.m[0][0] * m.m[1][1] * m.m[2][3] +
            m.m[0][0] * m.m[1][3] * m.m[2][1] +
            m.m[1][0] * m.m[0][1] * m.m[2][3] -
            m.m[1][0] * m.m[0][3] * m.m[2][1] -
            m.m[2][0] * m.m[0][1] * m.m[1][3] +
            m.m[2][0] * m.m[0][3] * m.m[1][1];

        inv.m[3][3] = m.m[0][0] * m.m[1][1] * m.m[2][2] -
            m.m[0][0] * m.m[1][2] * m.m[2][1] -
            m.m[1][0] * m.m[0][1] * m.m[2][2] +
            m.m[1][0] * m.m[0][2] * m.m[2][1] +
            m.m[2][0] * m.m[0][1] * m.m[1][2] -
            m.m[2][0] * m.m[0][2] * m.m[1][1];

        det = m.m[0][0] * inv.m[0][0] + m.m[0][1] * inv.m[1][0] + m.m[0][2] * inv.m[2][0] + m.m[0][3] * inv.m[3][0];

        if (det == 0.0)
            return Matrix4x4<T>(0);

        det = 1.0 / det;

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                inv.m[i][j] *= det;

        return inv;
    }

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> look_at(const Vec3f& eye, const Vec3f& look, const Vec3f& up)
    {
        Vec3<T> d = normalize(look - eye);
        Vec3<T> r = normalize(cross(d, up));
        Vec3<T> u = cross(r, d);

        Matrix4x4<T> out;

        out.m[0][0] = r.x;
        out.m[1][0] = r.y;
        out.m[2][0] = r.z;
        out.m[0][1] = u.x;
        out.m[1][1] = u.y;
        out.m[2][1] = u.z;
        out.m[0][2] = -d.x;
        out.m[1][2] = -d.y;
        out.m[2][2] = -d.z;
        out.m[3][0] = -dot(r, eye);
        out.m[3][1] = -dot(u, eye);
        out.m[3][2] = dot(d, eye);

        return out;
    }
    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> perspective(T fovy, T aspect, T zNear, T zFar)
    {
        float const tan_half_fovy = tan(fovy / 2.f);
        Matrix4x4<T> out;
        out.m[0][0] = 1.f / (aspect * tan_half_fovy);
        out.m[1][1] = 1.f / (tan_half_fovy);
        out.m[2][2] = -(zFar + zNear) / (zFar - zNear);
        out.m[2][3] = -1.f;
        out.m[3][2] = -(2.f * zFar * zNear) / (zFar - zNear);
        return out;
    }

    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> zero(void)
    {
        return Matrix4x4<T>(0);
    }
    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> ones(void)
    {
        return Matrix4x4<T>(1);
    }
    template<class T> _HOST_DEVICE
    inline Matrix4x4<T> unit(void)
    {
        return Matrix4x4<T>(
            1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    }

    /*
    inline __device__ dRay operator*(const Matrix4x4 m, const dRay r)
    {
        dRay out;

        Vec4f o = m * make_Vec4f(r.o.x, r.o.y, r.o.z, 1.f);
        Vec4f d = m * make_Vec4f(r.d.x, r.d.y, r.d.z, 0.f);

        out.o = make_Vec4f(o.x, o.y, o.z, 0.f);
        out.d = make_Vec4f(d.x, d.y, d.z, 0.f);

        return out;
    }
    inline __device__ dRay operator*(const dRay r, const Matrix4x4 m)
    {
        dRay out;

        Vec4f o = make_Vec4f(r.o.x, r.o.y, r.o.z, 1.f) * m;
        Vec4f d = make_Vec4f(r.d.x, r.d.y, r.d.z, 0.f) * m;

        out.o = make_Vec4f(o.x, o.y, o.z, 0.f);
        out.d = make_Vec4f(d.x, d.y, d.z, 0.f);

        return out;
    }

    static inline Matrix4x4 Matrix4x4_cast(const glm::mat4& m)
    {
        return Matrix4x4(
            m[0][0], m[0][1], m[0][2], m[0][3],
            m[1][0], m[1][1], m[1][2], m[1][3],
            m[2][0], m[2][1], m[2][2], m[2][3],
            m[3][0], m[3][1], m[3][2], m[3][3]
        );
    }
    */
}

#endif // _JEK_MATRIX_