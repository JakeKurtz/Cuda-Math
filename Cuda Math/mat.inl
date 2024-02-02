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

#include "mat.h"

namespace cml
{
    /* -------------------------------------------------------------------------- */
    /*                                 Comparators                                */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator==(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, U>& m2)
    {
        return
            m1.t00==m2.t00&&m1.t01==m2.t01&&m1.t02==m2.t02&&
            m1.t10==m2.t10&&m1.t11==m2.t11&&m1.t12==m2.t12&&
            m1.t20==m2.t20&&m1.t21==m2.t21&&m1.t22==m2.t22;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, U>& m2)
    {
        return
            m1.t00!=m2.t00&&m1.t01!=m2.t01&&m1.t02!=m2.t02&&
            m1.t10!=m2.t10&&m1.t11!=m2.t11&&m1.t12!=m2.t12&&
            m1.t20!=m2.t20&&m1.t21!=m2.t21&&m1.t22!=m2.t22;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator==(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, U>& m2)
    {
        return
            m1.t00==m2.t00&&m1.t01==m2.t01&&m1.t02==m2.t02&&
            m1.t10==m2.t10&&m1.t11==m2.t11&&m1.t12==m2.t12&&
            m1.t20==m2.t20&&m1.t21==m2.t21&&m1.t22==m2.t22;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, U>& m2)
    {
        return
            m1.t00!=m2.t00&&m1.t01!=m2.t01&&m1.t02!=m2.t02&&
            m1.t10!=m2.t10&&m1.t11!=m2.t11&&m1.t12!=m2.t12&&
            m1.t20!=m2.t20&&m1.t21!=m2.t21&&m1.t22!=m2.t22;
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator==(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, U>& m2)
    {
        return 
            m1.t00==m2.t00&&
            m1.t01==m2.t01&&
            m1.t10==m2.t10&&
            m1.t11==m2.t11;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T, class U>
    CML_FUNC_DECL CML_CONSTEXPR bool operator!=(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, U>& m2)
    {
        return 
            m1.t00!=m2.t00&&
            m1.t01!=m2.t01&&
            m1.t10!=m2.t10&&
            m1.t11!=m2.t11;
    }

    /* -------------------------------------------------------------------------- */
    /*                               Multiplication                               */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator*=(mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2)
    {
        mat4x4<T> tmp;

        tmp.t00 = m1.t00*m2.t00+m1.t01*m2.t10+m1.t02*m2.t20+m1.t03*m2.t30;
        tmp.t01 = m1.t00*m2.t01+m1.t01*m2.t11+m1.t02*m2.t21+m1.t03*m2.t31;
        tmp.t02 = m1.t00*m2.t02+m1.t01*m2.t12+m1.t02*m2.t22+m1.t03*m2.t32;
        tmp.t03 = m1.t00*m2.t03+m1.t01*m2.t13+m1.t02*m2.t23+m1.t03*m2.t33;

        tmp.t10 = m1.t10*m2.t00+m1.t11*m2.t10+m1.t12*m2.t20+m1.t13*m2.t30;
        tmp.t11 = m1.t10*m2.t01+m1.t11*m2.t11+m1.t12*m2.t21+m1.t13*m2.t31;
        tmp.t12 = m1.t10*m2.t02+m1.t11*m2.t12+m1.t12*m2.t22+m1.t13*m2.t32;
        tmp.t13 = m1.t10*m2.t03+m1.t11*m2.t13+m1.t12*m2.t23+m1.t13*m2.t33;

        tmp.t20 = m1.t20*m2.t00+m1.t21*m2.t10+m1.t22*m2.t20+m1.t23*m2.t30;
        tmp.t21 = m1.t20*m2.t01+m1.t21*m2.t11+m1.t22*m2.t21+m1.t23*m2.t31;
        tmp.t22 = m1.t20*m2.t02+m1.t21*m2.t12+m1.t22*m2.t22+m1.t23*m2.t32;
        tmp.t23 = m1.t20*m2.t03+m1.t21*m2.t13+m1.t22*m2.t23+m1.t23*m2.t33;

        tmp.t30 = m1.t30*m2.t00+m1.t31*m2.t10+m1.t32*m2.t20+m1.t33*m2.t30;
        tmp.t31 = m1.t30*m2.t01+m1.t31*m2.t11+m1.t32*m2.t21+m1.t33*m2.t31;
        tmp.t32 = m1.t30*m2.t02+m1.t31*m2.t12+m1.t32*m2.t22+m1.t33*m2.t32;
        tmp.t33 = m1.t30*m2.t03+m1.t31*m2.t13+m1.t32*m2.t23+m1.t33*m2.t33;

        m1 = tmp;
        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator*=(mat4x4_t<t_id, T>& m, const T s)
    {
        m.t00 *= s; m.t01 *= s; m.t02 *= s; m.t03 *= s;
        m.t10 *= s; m.t11 *= s; m.t12 *= s; m.t13 *= s;
        m.t20 *= s; m.t21 *= s; m.t22 *= s; m.t23 *= s;
        m.t30 *= s; m.t31 *= s; m.t32 *= s; m.t33 *= s;

        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec4_t<t_id, T> operator*=(vec4_t<t_id, T>& v, const mat4x4_t<u_id, T>& m)
    {
        vec4<T> tmp;

        tmp.x = v.x*m.t00+v.y*m.t01+v.z*m.t02+v.w*m.t03;
        tmp.y = v.x*m.t10+v.y*m.t11+v.z*m.t12+v.w*m.t13;
        tmp.z = v.x*m.t20+v.y*m.t21+v.z*m.t22+v.w*m.t23;
        tmp.w = v.x*m.t30+v.y*m.t31+v.z*m.t32+v.w*m.t33;

        v = tmp;
        return v;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator*(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2)
    {
        mat4x4<T> tmp(m1);
        return tmp *= m2;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator*(const vec4_t<t_id, T>& v, const mat4x4_t<u_id, T>& m)
    {
        vec4<T> tmp(v);
        return tmp *= m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec4<T> operator*(const mat4x4_t<t_id, T>& m, const vec4_t<u_id, T>& v)
    {
        vec4<T> tmp(v);
        return tmp *= transpose(m);
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator*(const mat4x4_t<t_id, T>& m, const T s)
    {
        mat4x4<T> tmp(m);
        return tmp *= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator*(const T s, const mat4x4_t<t_id, T>& m)
    {
        mat4x4<T> tmp(m);
        return tmp *= s;
    }

    /* --------------------------------- mat3x3 --------------------------------- */


    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator*=(mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2)
    {
        mat3x3<T> tmp;

        tmp.t00 = m1.t00*m2.t00+m1.t01*m2.t10+m1.t02*m2.t20;
        tmp.t01 = m1.t00*m2.t01+m1.t01*m2.t11+m1.t02*m2.t21;
        tmp.t02 = m1.t00*m2.t02+m1.t01*m2.t12+m1.t02*m2.t22;

        tmp.t10 = m1.t10*m2.t00+m1.t11*m2.t10+m1.t12*m2.t20;
        tmp.t11 = m1.t10*m2.t01+m1.t11*m2.t11+m1.t12*m2.t21;
        tmp.t12 = m1.t10*m2.t02+m1.t11*m2.t12+m1.t12*m2.t22;

        tmp.t20 = m1.t20*m2.t00+m1.t21*m2.t10+m1.t22*m2.t20;
        tmp.t21 = m1.t20*m2.t01+m1.t21*m2.t11+m1.t22*m2.t21;
        tmp.t22 = m1.t20*m2.t02+m1.t21*m2.t12+m1.t22*m2.t22;

        m1 = tmp;
        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator*=(mat3x3_t<t_id, T>& m, const T s)
    {
        m.t00 *= s; m.t01 *= s; m.t02 *= s;
        m.t10 *= s; m.t11 *= s; m.t12 *= s;
        m.t20 *= s; m.t21 *= s; m.t22 *= s;

        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec3_t<t_id, T> operator*=(vec3_t<t_id, T>& v, const mat3x3_t<u_id, T>& m)
    {
        vec3<T> tmp;

        tmp.x = v.x*m.t00+v.y*m.t01+v.z*m.t02;
        tmp.y = v.x*m.t10+v.y*m.t11+v.z*m.t12;
        tmp.z = v.x*m.t20+v.y*m.t21+v.z*m.t22;

        v = tmp;
        return v;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator*(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2)
    {
        mat3x3<T> tmp(m1);
        return tmp *= m2;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator*(const vec3_t<t_id, T>& v, const mat3x3_t<u_id, T>& m)
    {
        vec3<T> tmp(v);
        return tmp *= m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec3<T> operator*(const mat3x3_t<t_id, T>& m, const vec3_t<u_id, T>& v)
    {
        vec3<T> tmp(v);
        return tmp *= transpose(m);
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator*(const mat3x3_t<t_id, T>& m, const T s)
    {
        mat3x3<T> tmp(m);
        return tmp *= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator*(const T s, const mat3x3_t<t_id, T>& m)
    {
        mat3x3<T> tmp(m);
        return tmp *= s;
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator*=(mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2)
    {
        mat2x2<T> tmp;
        tmp.t00 = m1.t00*m2.t00+m1.t01*m2.t10;
        tmp.t01 = m1.t00*m2.t01+m1.t01*m2.t11;
        tmp.t10 = m1.t10*m2.t00+m1.t11*m2.t10;
        tmp.t11 = m1.t10*m2.t01+m1.t11*m2.t11;

        m1 = tmp;
        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator*=(mat2x2_t<t_id, T>& m, const T s)
    {
        /*
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                m(i, j) *= s;
        return m;
        */
        m.t00 *= s; m.t01 *= s;
        m.t10 *= s; m.t11 *= s;
        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec2_t<t_id, T> operator*=(vec2_t<t_id, T>& v, const mat2x2_t<u_id, T>& m)
    {
        vec2<T> tmp;
        tmp.x = v.x*m.t00+v.y*m.t01;
        tmp.y = v.x*m.t10+v.y*m.t11;

        v = tmp;
        return v;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator*(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2)
    {
        mat2x2<T> tmp(m1);
        return tmp *= m2;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator*(const vec2_t<t_id, T>& v, const mat2x2_t<u_id, T>& m)
    {
        vec2<T> tmp(v);
        return tmp *= m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR vec2<T> operator*(const mat2x2_t<t_id, T>& m, const vec2_t<u_id, T>& v)
    {
        vec2<T> tmp(v);
        return tmp *= transpose(m);
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator*(const mat2x2_t<t_id, T>& m, const T s)
    {
        mat2x2<T> tmp(m);
        return tmp *= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator*(const T s, const mat2x2_t<t_id, T>& m)
    {
        mat2x2<T> tmp(m);
        return tmp *= s;
    }

    /* -------------------------------------------------------------------------- */
    /*                                  Division                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator/=(mat4x4_t<t_id, T>& m, const T s)
    {
        m.t00 /= s; m.t01 /= s; m.t02 /= s; m.t03 /= s;
        m.t10 /= s; m.t11 /= s; m.t12 /= s; m.t13 /= s;
        m.t20 /= s; m.t21 /= s; m.t22 /= s; m.t23 /= s;
        m.t30 /= s; m.t31 /= s; m.t32 /= s; m.t33 /= s;

        return m;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator/(const mat4x4_t<t_id, T>& m, const T s)
    {
        mat4x4<T> tmp(m);
        return tmp /= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator/(const T s, const mat4x4_t<t_id, T>& m)
    {
        mat4x4<T> tmp(m);

        tmp.t00 = s/m.t00; tmp.t01 = s/m.t01; tmp.t02 = s/m.t02; tmp.t03 = s/m.t03;
        tmp.t10 = s/m.t10; tmp.t11 = s/m.t11; tmp.t12 = s/m.t12; tmp.t13 = s/m.t13;
        tmp.t20 = s/m.t20; tmp.t21 = s/m.t21; tmp.t22 = s/m.t22; tmp.t23 = s/m.t23;
        tmp.t30 = s/m.t30; tmp.t31 = s/m.t31; tmp.t32 = s/m.t32; tmp.t33 = s/m.t33;

        return tmp;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator/=(mat3x3_t<t_id, T>& m, const T s)
    {
        m.t00 /= s; m.t01 /= s; m.t02 /= s;
        m.t10 /= s; m.t11 /= s; m.t12 /= s;
        m.t20 /= s; m.t21 /= s; m.t22 /= s;

        return m;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator/(const mat3x3_t<t_id, T>& m, const T s)
    {
        mat3x3<T> tmp(m);
        return tmp /= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator/(const T s, const mat3x3_t<t_id, T>& m)
    {
        mat3x3<T> tmp(m);

        tmp.t00 = s/m.t00; tmp.t01 = s/m.t01; tmp.t02 = s/m.t02;
        tmp.t10 = s/m.t10; tmp.t11 = s/m.t11; tmp.t12 = s/m.t12;
        tmp.t20 = s/m.t20; tmp.t21 = s/m.t21; tmp.t22 = s/m.t22;

        return tmp;
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator/=(mat2x2_t<t_id, T>& m, const T s)
    {
        m.t00 /= s; m.t01 /= s;
        m.t10 /= s; m.t11 /= s;
        return m;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator/(const mat2x2_t<t_id, T>& m, const T s)
    {
        mat2x2<T> tmp(m);
        return tmp /= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator/(const T s, const mat2x2_t<t_id, T>& m)
    {
        mat2x2<T> tmp(m);
        tmp.t00 = s/m.t00; tmp.t01 = s/m.t01;
        tmp.t10 = s/m.t10; tmp.t11 = s/m.t11;
        return tmp;
    }

    /* -------------------------------------------------------------------------- */
    /*                                  Addition                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator+=(mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2)
    {
        m1.t00 += m2.t00; m1.t01 += m2.t01; m1.t02 += m2.t02; m1.t03 += m2.t03;
        m1.t10 += m2.t10; m1.t11 += m2.t11; m1.t12 += m2.t12; m1.t13 += m2.t13;
        m1.t20 += m2.t20; m1.t21 += m2.t21; m1.t22 += m2.t22; m1.t23 += m2.t23;
        m1.t30 += m2.t30; m1.t31 += m2.t31; m1.t32 += m2.t32; m1.t33 += m2.t33;

        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator+=(mat4x4_t<t_id, T>& m, const T s)
    {
        m.t00 += s; m.t01 += s; m.t02 += s; m.t03 += s;
        m.t10 += s; m.t11 += s; m.t12 += s; m.t13 += s;
        m.t20 += s; m.t21 += s; m.t22 += s; m.t23 += s;
        m.t30 += s; m.t31 += s; m.t32 += s; m.t33 += s;

        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator+(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2)
    {
        mat4x4<T> tmp(m1);
        return tmp += m2;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator+(const mat4x4_t<t_id, T>& m, const T s)
    {
        mat4x4<T> tmp(m);
        return tmp += s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator+(const T s, const mat4x4_t<t_id, T>& m)
    {
        mat4x4<T> tmp(m);
        return tmp += s;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator+=(mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2)
    {
        m1.t00 += m2.t00; m1.t01 += m2.t01; m1.t02 += m2.t02;
        m1.t10 += m2.t10; m1.t11 += m2.t11; m1.t12 += m2.t12;
        m1.t20 += m2.t20; m1.t21 += m2.t21; m1.t22 += m2.t22;

        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator+=(mat3x3_t<t_id, T>& m, const T s)
    {
        m.t00 += s; m.t01 += s; m.t02 += s;
        m.t10 += s; m.t11 += s; m.t12 += s;
        m.t20 += s; m.t21 += s; m.t22 += s;

        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator+(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2)
    {
        mat3x3<T> tmp(m1);
        return tmp += m2;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator+(const mat3x3_t<t_id, T>& m, const T s)
    {
        mat3x3<T> tmp(m);
        return tmp += s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator+(const T s, const mat3x3_t<t_id, T>& m)
    {
        mat3x3<T> tmp(m);
        return tmp += s;
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator+=(mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2)
    {
        m1.t00 += m2.t00; m1.t01 += m2.t01;
        m1.t10 += m2.t10; m1.t11 += m2.t11;
        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator+=(mat2x2_t<t_id, T>& m, const T s)
    {
        m.t00 += s; m.t01 += s;
        m.t10 += s; m.t11 += s;
        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator+(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2)
    {
        mat2x2<T> tmp(m1);
        return tmp += m2;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator+(const mat2x2_t<t_id, T>& m, const T s)
    {
        mat2x2<T> tmp(m);
        return tmp += s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator+(const T s, const mat2x2_t<t_id, T>& m)
    {
        mat2x2<T> tmp(m);
        return tmp += s;
    }

    /* -------------------------------------------------------------------------- */
    /*                                 Subtraction                                */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator-=(mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2)
    {
        m1.t00 -= m2.t00; m1.t01 -= m2.t01; m1.t02 -= m2.t02; m1.t03 -= m2.t03;
        m1.t10 -= m2.t10; m1.t11 -= m2.t11; m1.t12 -= m2.t12; m1.t13 -= m2.t23;
        m1.t20 -= m2.t20; m1.t21 -= m2.t21; m1.t22 -= m2.t22; m1.t23 -= m2.t23;
        m1.t30 -= m2.t30; m1.t31 -= m2.t31; m1.t32 -= m2.t32; m1.t33 -= m2.t33;

        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator-=(mat4x4_t<t_id, T>& m, const T s)
    {
        m.t00 -= s; m.t01 -= s; m.t02 -= s; m.t03 -= s;
        m.t10 -= s; m.t11 -= s; m.t12 -= s; m.t13 -= s;
        m.t20 -= s; m.t21 -= s; m.t22 -= s; m.t23 -= s;
        m.t30 -= s; m.t31 -= s; m.t32 -= s; m.t33 -= s;

        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator-(const mat4x4_t<t_id, T>& m1, const mat4x4_t<u_id, T>& m2)
    {
        mat4x4<T> tmp(m1);
        return tmp -= m2;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator-(const mat4x4_t<t_id, T>& m, const T s)
    {
        mat4x4<T> tmp(m);
        return tmp -= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4<T> operator-(const T s, const mat4x4_t<t_id, T>& m)
    {
        mat4x4<T> tmp(-m);
        return tmp += s;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator-=(mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2)
    {
        m1.t00 -= m2.t00; m1.t01 -= m2.t01; m1.t02 -= m2.t02;
        m1.t10 -= m2.t10; m1.t11 -= m2.t11; m1.t12 -= m2.t12;
        m1.t20 -= m2.t20; m1.t21 -= m2.t21; m1.t22 -= m2.t22;

        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator-=(mat3x3_t<t_id, T>& m, const T s)
    {
        m.t00 -= s; m.t01 -= s; m.t02 -= s;
        m.t10 -= s; m.t11 -= s; m.t12 -= s;
        m.t20 -= s; m.t21 -= s; m.t22 -= s;

        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator-(const mat3x3_t<t_id, T>& m1, const mat3x3_t<u_id, T>& m2)
    {
        mat3x3<T> tmp(m1);
        return tmp -= m2;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator-(const mat3x3_t<t_id, T>& m, const T s)
    {
        mat3x3<T> tmp(m);
        return tmp -= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3<T> operator-(const T s, const mat3x3_t<t_id, T>& m)
    {
        mat3x3<T> tmp(-m);
        return tmp += s;
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator-=(mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2)
    {
        m1.t00 -= m2.t00; m1.t01 -= m2.t01;
        m1.t10 -= m2.t10; m1.t11 -= m2.t11;
        return m1;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator-=(mat2x2_t<t_id, T>& m, const T s)
    {
        m.t00 += s; m.t01 += s;
        m.t10 += s; m.t11 += s;
        return m;
    }

    template <ASX::ID t_id, ASX::ID u_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator-(const mat2x2_t<t_id, T>& m1, const mat2x2_t<u_id, T>& m2)
    {
        mat2x2<T> tmp(m1);
        return tmp -= m2;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator-(const mat2x2_t<t_id, T>& m, const T s)
    {
        mat2x2<T> tmp(m);
        return tmp -= s;
    }

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2<T> operator-(const T s, const mat2x2_t<t_id, T>& m)
    {
        mat2x2<T> tmp(-m);
        return tmp += s;
    }

    /* -------------------------------------------------------------------------- */
    /*                                  Negation                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> operator-(const mat4x4_t<t_id, T>& m)
    {
        mat4x4<T> tmp(m);

        tmp.t00 = -tmp.t00; tmp.t01 = -tmp.t01; tmp.t02 = -tmp.t02; tmp.t03 = -tmp.t03;
        tmp.t10 = -tmp.t10; tmp.t11 = -tmp.t11; tmp.t12 = -tmp.t12; tmp.t13 = -tmp.t13;
        tmp.t20 = -tmp.t20; tmp.t21 = -tmp.t21; tmp.t22 = -tmp.t22; tmp.t23 = -tmp.t23;
        tmp.t30 = -tmp.t30; tmp.t31 = -tmp.t31; tmp.t32 = -tmp.t32; tmp.t33 = -tmp.t33;

        return m;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> operator-(const mat3x3_t<t_id, T>& m)
    {
        mat3x3<T> tmp(m);

        tmp.t00 = -tmp.t00; tmp.t01 = -tmp.t01; tmp.t02 = -tmp.t02;
        tmp.t10 = -tmp.t10; tmp.t11 = -tmp.t11; tmp.t12 = -tmp.t12;
        tmp.t20 = -tmp.t20; tmp.t21 = -tmp.t21; tmp.t22 = -tmp.t22;

        return m;
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> operator-(const mat2x2_t<t_id, T>& m)
    {
        mat2x2<T> tmp(m);
        tmp.t00 = -tmp.t00; tmp.t01 = -tmp.t01;
        tmp.t10 = -tmp.t10; tmp.t11 = -tmp.t11;
        return m;
    }

    /* -------------------------------------------------------------------------- */
    /*                                  Transpose                                 */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> transpose(const mat4x4_t<t_id, T>& m)
    {
        mat4x4<T> tmp(
            m.t00, m.t10, m.t20, m.t30,
            m.t01, m.t11, m.t21, m.t31,
            m.t02, m.t12, m.t22, m.t32,
            m.t03, m.t13, m.t23, m.t33);
        return tmp;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> transpose(const mat3x3_t<t_id, T>& m)
    {
        mat3x3<T> tmp(
            m.t00, m.t10, m.t20,
            m.t01, m.t11, m.t21,
            m.t02, m.t12, m.t22);
        return tmp;
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> transpose(const mat2x2_t<t_id, T>& m)
    {
        mat2x2<T> tmp(
            m.t00, m.t10,
            m.t01, m.t11);
        return tmp;
    }

    /* -------------------------------------------------------------------------- */
    /*                                 Determinant                                */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR T det(const mat4x4_t<t_id, T>& m)
    {
        T d =
            m.t30*m.t21*m.t12*m.t03-m.t20*m.t31*m.t12*m.t03-
            m.t30*m.t11*m.t22*m.t03+m.t10*m.t31*m.t22*m.t03+
            m.t20*m.t11*m.t32*m.t03-m.t10*m.t21*m.t32*m.t03-
            m.t30*m.t21*m.t02*m.t13+m.t20*m.t31*m.t02*m.t13+
            m.t30*m.t01*m.t22*m.t13-m.t00*m.t31*m.t22*m.t13-
            m.t20*m.t01*m.t32*m.t13+m.t00*m.t21*m.t32*m.t13+
            m.t30*m.t11*m.t02*m.t23-m.t10*m.t31*m.t02*m.t23-
            m.t30*m.t01*m.t12*m.t23+m.t00*m.t31*m.t12*m.t23+
            m.t10*m.t01*m.t32*m.t23-m.t00*m.t11*m.t32*m.t23-
            m.t20*m.t11*m.t02*m.t33+m.t10*m.t21*m.t02*m.t33+
            m.t20*m.t01*m.t12*m.t33-m.t00*m.t21*m.t12*m.t33-
            m.t10*m.t01*m.t22*m.t33+m.t00*m.t11*m.t22*m.t33;

        return d;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR T det(const mat3x3_t<t_id, T>& m)
    {
        return
            m.t00*(m.t11*m.t22-m.t21*m.t12);
        -m.t01*(m.t10*m.t22-m.t20*m.t12);
        +m.t02*(m.t10*m.t21-m.t20*m.t11);
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR T det(const mat2x2_t<t_id, T>& m)
    {
        return m.t00*m.t11-m.t01*m.t10;
    }

    /* -------------------------------------------------------------------------- */
    /*                                   Inverse                                  */
    /* -------------------------------------------------------------------------- */

    /* --------------------------------- mat4x4 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat4x4_t<t_id, T> inv(const mat4x4_t<t_id, T>& m)
    {
        mat4x4<T> tmp;

        tmp.t00 = m.t11*m.t22*m.t33-
            m.t11*m.t23*m.t32-
            m.t21*m.t12*m.t33+
            m.t21*m.t13*m.t32+
            m.t31*m.t12*m.t23-
            m.t31*m.t13*m.t22;

        tmp.t10 = -m.t10*m.t22*m.t33+
            m.t10*m.t23*m.t32+
            m.t20*m.t12*m.t33-
            m.t20*m.t13*m.t32-
            m.t30*m.t12*m.t23+
            m.t30*m.t13*m.t22;

        tmp.t20 = m.t10*m.t21*m.t33-
            m.t10*m.t23*m.t31-
            m.t20*m.t11*m.t33+
            m.t20*m.t13*m.t31+
            m.t30*m.t11*m.t23-
            m.t30*m.t13*m.t21;

        tmp.t30 = -m.t10*m.t21*m.t32+
            m.t10*m.t22*m.t31+
            m.t20*m.t11*m.t32-
            m.t20*m.t12*m.t31-
            m.t30*m.t11*m.t22+
            m.t30*m.t12*m.t21;

        tmp.t01 = -m.t01*m.t22*m.t33+
            m.t01*m.t23*m.t32+
            m.t21*m.t02*m.t33-
            m.t21*m.t03*m.t32-
            m.t31*m.t02*m.t23+
            m.t31*m.t03*m.t22;

        tmp.t11 = m.t00*m.t22*m.t33-
            m.t00*m.t23*m.t32-
            m.t20*m.t02*m.t33+
            m.t20*m.t03*m.t32+
            m.t30*m.t02*m.t23-
            m.t30*m.t03*m.t22;

        tmp.t21 = -m.t00*m.t21*m.t33+
            m.t00*m.t23*m.t31+
            m.t20*m.t01*m.t33-
            m.t20*m.t03*m.t31-
            m.t30*m.t01*m.t23+
            m.t30*m.t03*m.t21;

        tmp.t31 = m.t00*m.t21*m.t32-
            m.t00*m.t22*m.t31-
            m.t20*m.t01*m.t32+
            m.t20*m.t02*m.t31+
            m.t30*m.t01*m.t22-
            m.t30*m.t02*m.t21;

        tmp.t02 = m.t01*m.t12*m.t33-
            m.t01*m.t13*m.t32-
            m.t11*m.t02*m.t33+
            m.t11*m.t03*m.t32+
            m.t31*m.t02*m.t13-
            m.t31*m.t03*m.t12;

        tmp.t12 = -m.t00*m.t12*m.t33+
            m.t00*m.t13*m.t32+
            m.t10*m.t02*m.t33-
            m.t10*m.t03*m.t32-
            m.t30*m.t02*m.t13+
            m.t30*m.t03*m.t12;

        tmp.t22 = m.t00*m.t11*m.t33-
            m.t00*m.t13*m.t31-
            m.t10*m.t01*m.t33+
            m.t10*m.t03*m.t31+
            m.t30*m.t01*m.t13-
            m.t30*m.t03*m.t11;

        tmp.t32 = -m.t00*m.t11*m.t32+
            m.t00*m.t12*m.t31+
            m.t10*m.t01*m.t32-
            m.t10*m.t02*m.t31-
            m.t30*m.t01*m.t12+
            m.t30*m.t02*m.t11;

        tmp.t03 = -m.t01*m.t12*m.t23+
            m.t01*m.t13*m.t22+
            m.t11*m.t02*m.t23-
            m.t11*m.t03*m.t22-
            m.t21*m.t02*m.t13+
            m.t21*m.t03*m.t12;

        tmp.t13 = m.t00*m.t12*m.t23-
            m.t00*m.t13*m.t22-
            m.t10*m.t02*m.t23+
            m.t10*m.t03*m.t22+
            m.t20*m.t02*m.t13-
            m.t20*m.t03*m.t12;

        tmp.t23 = -m.t00*m.t11*m.t23+
            m.t00*m.t13*m.t21+
            m.t10*m.t01*m.t23-
            m.t10*m.t03*m.t21-
            m.t20*m.t01*m.t13+
            m.t20*m.t03*m.t11;

        tmp.t33 = m.t00*m.t11*m.t22-
            m.t00*m.t12*m.t21-
            m.t10*m.t01*m.t22+
            m.t10*m.t02*m.t21+
            m.t20*m.t01*m.t12-
            m.t20*m.t02*m.t11;

        T det = m.t00*tmp.t00+m.t01*tmp.t10+m.t02*tmp.t20+m.t03*tmp.t30;

        if (det==0.0) {}

        det = 1.0/det;
        tmp *= det;

        return tmp;
    }

    /* --------------------------------- mat3x3 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat3x3_t<t_id, T> inv(const mat3x3_t<t_id, T>& m)
    {
        T _det = static_cast<T>(1)/det(m);
        if (_det==static_cast<T>(0)) {
            // log an error
            return NAN;
        }
        return _det*mat3x3<T>(1.0);
    }

    /* --------------------------------- mat2x2 --------------------------------- */

    template <ASX::ID t_id, class T>
    CML_FUNC_DECL CML_CONSTEXPR mat2x2_t<t_id, T> inv(const mat2x2_t<t_id, T>& m)
    {
        T _det = static_cast<T>(1)/det(m);
        if (_det==static_cast<T>(0)) {
            // log an error
            return NAN;
        }
        return _det*mat2x2<T>(m.t11, -m.t01, -m.t10, m.t00);
    }

} // namespace cml
