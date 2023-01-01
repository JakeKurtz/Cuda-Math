#ifndef ASA_HPP
#define ASA_HPP
//------------------------------------------------------
// ASX : ASA.hpp : $Name: release_2_1_0 $
// Copyright (C) 2010, 2014, Robert Strzodka
// License: GPLv3 or later
//------------------------------------------------------
//
// Array of Structs of Arrays (ASA) utlility classes
//
//------------------------------------------------------
//
// This file is part of ASX.
// 
// ASX is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ASX is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ASX. If not, see <http://www.gnu.org/licenses/>.
//
//------------------------------------------------------

#include <cstddef>
#include <climits>
#include <cassert>

//------------------------------------------------------
// define CUDA qualifiers
//------------------------------------------------------

#if defined(__CUDACC__)
#include <cuda.h>
#include <helper_cuda.h>
#else
#define __host__
#define __device__
#endif

//======================================================
namespace ASX {
    //======================================================

    //------------------------------------------------------
    // util
    //------------------------------------------------------

    template <bool cond, class Type1, class Type2> struct StaticIfElse {};
    template <class Type1, class Type2> struct StaticIfElse<true, Type1, Type2> { typedef Type1 Type; };
    template <class Type1, class Type2> struct StaticIfElse<false, Type1, Type2> { typedef Type2 Type; };

    //------------------------------------------------------
    // enum
    //------------------------------------------------------

    enum Layout {
        AOS,
        SOA,
        LAYOUT_NUM
    };
    static const char* getLayoutStr(Layout tag)
    {
        const char* str[LAYOUT_NUM] = { "AoS", "SoA" };
        assert(tag >= 0 && tag < LAYOUT_NUM); return str[tag];
    }

    // Some compiler treat all enums as ints so we cannot use the
    // enum type directly as the template parameter type
    enum /* ID */ {
        ID_value = -10,
        ID_one = 1,
        ID_array
    };

    //------------------------------------------------------
    // common types
    //------------------------------------------------------

    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;
    typedef difference_type ID;


    //======================================================
    // Array
    //======================================================
    //
    template <ASX::Layout t_layout, ASX::size_type t_maxSize, class TT_FlexType>
    class Array {};

#define ASX_Array_common_types \
  typedef       ValueType                                                                 value_type; \
  typedef       AccessType*                                                               pointer; \
  typedef const AccessType*                                                               const_pointer; \
  typedef       AccessType&                                                               reference; \
  typedef const AccessType&                                                               const_reference; \
  typedef typename StaticIfElse<(s_maxSize<=INT_MAX), unsigned int, ASX::size_type>::Type size_type; \
  typedef typename StaticIfElse<(s_maxSize<=INT_MAX), int, ASX::difference_type>::Type    difference_type;

#define ASX_Array_common_functions \
  /* element access */ \
  __host__            const_reference at         (difference_type index) const { assert(index>=0 && index<size()); return operator[](index); } \
  __host__                  reference at         (difference_type index)       { assert(index>=0 && index<size()); return operator[](index); } \
  __host__ __device__ const_reference front      ()                      const { return operator[](0); } \
  __host__ __device__       reference front      ()                            { return operator[](0); } \
  __host__ __device__ const_reference back       ()                      const { return operator[](size()-1); } \
  __host__ __device__       reference back       ()                            { return operator[](size()-1); } \
  /* capacity */ \
  __host__ __device__ size_type       size       ()                      const { return s_maxSize; } \
  __host__ __device__ size_type       capacity   ()                      const { return s_maxSize; } \
  __host__ __device__ size_type       max_size   ()                      const { return s_maxSize; } \
  __host__ __device__ bool            empty      ()                      const { return (size()==0); } \
  __host__ __device__ size_type       memory_size()                      const { return sizeof(*this); }

    //------------------------------------------------------
    // Array: AoS specialization
    //------------------------------------------------------

    template <ASX::size_type t_maxSize, template <ASX::ID, typename...> class TT_FlexType, typename ...Ts>
    class Array<ASX::AOS, t_maxSize, TT_FlexType<ID_value, Ts...>> {
    public:
        static const ASX::Layout    s_layout = ASX::AOS;
        static const ASX::size_type s_maxSize = t_maxSize;
        static const int            s_groupMemSize = sizeof(TT_FlexType<ID_value, Ts...>);
        static const int            s_groupNum = 1;

        typedef TT_FlexType<ID_value, Ts...>                   ValueType;
        typedef TT_FlexType<ID_value, Ts...>                   AccessType;
        ASX_Array_common_types

        __host__ __device__ const_reference operator[] (difference_type index) const { return arr[index]; }
        __host__ __device__       reference operator[] (difference_type index) { return arr[index]; }
        __host__ __device__ const void* data()                      const { return arr; }
        __host__ __device__       void* data() { return arr; }
        ASX_Array_common_functions

    protected:
        ValueType arr[s_maxSize];
    };

    //------------------------------------------------------
    // Array: SoA specialization
    //------------------------------------------------------

    template <ASX::size_type t_maxSize, template <ASX::ID, typename...> class TT_FlexType, typename ...Ts>
    class Array<ASX::SOA, t_maxSize, TT_FlexType<ID_value, Ts...>> {
    public:
        static const ASX::Layout    s_layout = ASX::SOA;
        static const ASX::size_type s_maxSize = t_maxSize;
        static const int            s_groupMemSize = sizeof(typename TT_FlexType<ID_one, Ts...>::ASX_ASA);
        static const int            s_groupNum = sizeof(TT_FlexType<ID_one, Ts...>) / s_groupMemSize;

        typedef TT_FlexType<ID_value, Ts...>                         ValueType;
        typedef TT_FlexType<static_cast<ASX::ID>(s_maxSize), Ts...>  AccessType;
        ASX_Array_common_types

        __host__ __device__ const_reference operator[] (difference_type index) const { return reinterpret_cast<const_reference>(arr[index]); }
        __host__ __device__       reference operator[] (difference_type index) { return reinterpret_cast<reference>(arr[index]); }
        __host__ __device__ const void* data()                      const { return arr; }
        __host__ __device__       void* data() { return arr; }
        ASX_Array_common_functions

    protected:
        char arr[s_groupNum * s_maxSize][s_groupMemSize];
    };


    //======================================================
    // Vector
    //======================================================

    template <ASX::Layout t_layout, ASX::size_type t_grainSize, class TT_FlexType>
    class Vector {};

#define ASX_Vector_common_functions \
  /* element access */ \
  __host__            const_reference at         (difference_type index) const { assert(index>=0 && index<size()); return operator[](index); } \
  __host__                  reference at         (difference_type index)       { assert(index>=0 && index<size()); return operator[](index); } \
  __host__ __device__ const_reference front      ()                      const { return operator[](0); } \
  __host__ __device__       reference front      ()                            { return operator[](0); } \
  __host__ __device__ const_reference back       ()                      const { return operator[](size()-1); } \
  __host__ __device__       reference back       ()                            { return operator[](size()-1); } \
  /* capacity */ \
  __host__ __device__ size_type       size       ()                      const { return curSize; } \
  __host__ __device__ size_type       capacity   ()                      const { return allocSize; } \
  __host__ __device__ size_type       max_size   ()                      const { return s_maxSize; } \
  __host__ __device__ bool            empty      ()                      const { return (size()==0); } \
  __host__ __device__ size_type       memory_size()                      const { return (allocSize/s_grainSize)*sizeof(grain_type); }

    //------------------------------------------------------
    // deep copy of Vector between host and GPU
    //------------------------------------------------------

#if defined(__CUDACC__)
    template <class T_Vector>
    void deepCopyVector(void* pDstObj, void* pDstData, const T_Vector* pSrc, enum cudaMemcpyKind kind)
    {
        assert(pDstObj && pSrc);

        T_Vector tmp, * pDst;
        switch (kind) {
        case cudaMemcpyHostToDevice:
            tmp = *pSrc;
            tmp.relocate(pDstData);
            checkCudaErrors(cudaMemcpy(pDstObj, &tmp, sizeof(T_Vector), kind));
            if (pDstData) checkCudaErrors(cudaMemcpy(pDstData, pSrc->data(), pSrc->memory_size(), kind));
            break;
        case cudaMemcpyDeviceToHost:
        case cudaMemcpyHostToHost:
            checkCudaErrors(cudaMemcpy(pDstObj, pSrc, sizeof(T_Vector), kind));
            pDst = static_cast<T_Vector*>(pDstObj);
            if (pDstData) checkCudaErrors(cudaMemcpy(pDstData, pDst->data(), pDst->memory_size(), kind));
            pDst->relocate(pDstData);
            break;
        case cudaMemcpyDeviceToDevice:
            checkCudaErrors(cudaMemcpy(&tmp, pSrc, sizeof(T_Vector), cudaMemcpyDeviceToHost));
            if (pDstData) checkCudaErrors(cudaMemcpy(pDstData, tmp.data(), tmp.memory_size(), kind));
            tmp.relocate(pDstData);
            checkCudaErrors(cudaMemcpy(pDstObj, &tmp, sizeof(T_Vector), cudaMemcpyHostToDevice));
            break;
        default:
            checkCudaErrors(cudaErrorInvalidValue);
        }
        tmp.relocate(0); // set tmp to 0 or else the dtor will try to delete someone's data
    }
#else
    template <class T_Vector>
    void deepCopyVector(void* pDstObj, void* pDstData, const T_Vector* pSrc)
    {
        {
            memcpy(pDstObj, pSrc, sizeof(T_Vector));
            pDst = static_cast<T_Vector*>(pDstObj);
            if (pDstData) memcpy(pDstData, pDst->data(), pDst->memory_size());
            pDst->relocate(pDstData);
        }
    }
#endif

    //------------------------------------------------------
    // Vector: AoS specialization
    //------------------------------------------------------

    template <ASX::size_type t_grainSize, template <ASX::ID, typename...> class TT_FlexType, typename ...Ts>
    class Vector<ASX::AOS, t_grainSize, TT_FlexType<ID_value, Ts...>> {
    public:
        static const ASX::Layout    s_layout = ASX::AOS;
        static const ASX::size_type s_maxSize = ULLONG_MAX;
        static const ASX::size_type s_grainSize = (t_grainSize <= 0) ? 1 : t_grainSize;
        static const int            s_groupMemSize = sizeof(TT_FlexType<ID_value, Ts...>);
        static const int            s_groupNum = 1;

        typedef       TT_FlexType<ID_value, Ts...>          value_type;
        typedef       TT_FlexType<ID_value, Ts...>          grain_type[s_grainSize];
        typedef       TT_FlexType<ID_value, Ts...>*         pointer;
        typedef const TT_FlexType<ID_value, Ts...>*         const_pointer;
        typedef       TT_FlexType<ID_value, Ts...>&         reference;
        typedef const TT_FlexType<ID_value, Ts...>&         const_reference;
        typedef       ASX::size_type                        size_type;
        typedef       ASX::difference_type                  difference_type;

        Vector(size_type n = 0) :
            curSize(n), allocSize(s_applyGrain(n)), base(allocSize ? new grain_type[allocSize / s_grainSize] : 0),
            hdl(reinterpret_cast<value_type*>(base)) {}
        ~Vector() { delete[] base; }

        __host__ __device__ const_reference operator[] (difference_type index) const { return hdl[index]; }
        __host__ __device__       reference operator[] (difference_type index) { return hdl[index]; }
        __host__ __device__ const void* data()                      const { return base; }
        __host__ __device__       void* data() { return base; }
        ASX_Vector_common_functions

            __host__ __device__ static size_type s_applyGrain(size_type n) { return (n + s_grainSize - 1) / s_grainSize * s_grainSize; }
        __host__ __device__ void relocate(void* pos) { base = static_cast<grain_type*>(pos); hdl = reinterpret_cast<value_type*>(base); }

    protected:
        size_type curSize, allocSize;
        grain_type* base;
        value_type* hdl;
    };

    //------------------------------------------------------
    // Vector: SoA specialization
    //------------------------------------------------------

    template <ASX::size_type t_grainSize, template <ASX::ID, typename...> class TT_FlexType, typename ...Ts>
    class Vector<ASX::SOA, t_grainSize, TT_FlexType<ID_value, Ts...>> {
    public:
        static const ASX::Layout    s_layout = ASX::SOA;
        static const ASX::size_type s_maxSize = ULLONG_MAX;
        static const ASX::size_type s_grainSize = (t_grainSize <= 0) ? 32 : (t_grainSize + 15) / 16 * 16;
        static const int            s_groupMemSize = sizeof(typename TT_FlexType<ID_one, Ts...>::ASX_ASA);
        static const int            s_groupNum = sizeof(TT_FlexType<ID_one, Ts...>) / s_groupMemSize;

        typedef       TT_FlexType<ID_value, Ts...>                          value_type;
        typedef       TT_FlexType<static_cast<ASX::ID>(s_grainSize), Ts...> grain_type;
        typedef       char                                                  group_type[s_groupMemSize];
        typedef       grain_type* pointer;
        typedef const grain_type* const_pointer;
        typedef       grain_type& reference;
        typedef const grain_type& const_reference;
        typedef       ASX::size_type                                        size_type;
        typedef       ASX::difference_type                                  difference_type;

        Vector(size_type n = 0) :
            curSize(n), allocSize(s_applyGrain(n)), base(allocSize ? new grain_type[allocSize / s_grainSize] : 0),
            hdl(reinterpret_cast<group_type*>(base)) {}
        ~Vector() { delete[] base; }

        __host__ __device__ const_reference operator[] (difference_type index) const { return reinterpret_cast<const_reference>(hdl[index * s_groupNum - (s_groupNum - 1) * (index % s_grainSize)]); }
        __host__ __device__       reference operator[] (difference_type index) { return reinterpret_cast<reference>(hdl[index * s_groupNum - (s_groupNum - 1) * (index % s_grainSize)]); }
        __host__ __device__ const void* data()                      const { return base; }
        __host__ __device__       void* data() { return base; }
        ASX_Vector_common_functions

            __host__ __device__ static size_type s_applyGrain(size_type n) { return (n + s_grainSize - 1) / s_grainSize * s_grainSize; }
        __host__ __device__ void relocate(void* pos) { base = static_cast<grain_type*>(pos); hdl = reinterpret_cast<group_type*>(base); }

    protected:
        size_type curSize, allocSize;
        grain_type* base;
        group_type* hdl;
    };


    //======================================================
    // ASAGroup
    //======================================================

    template <class T_Type, ASX::ID t_id> class ASAGroup { T_Type dummy[t_id]; };
    template <class T_Type> class ASAGroup<T_Type, ID_value> { char dummy; };


    //======================================================
    // FlexibleArray
    //======================================================

    template <class T_Element, ASX::size_type t_maxSize, ASX::size_type t_groupSize = 1>
    class FlexibleArray {
    public:

        template <ASX::ID t_id = ID_value> class TTypeASA {
        public:
            static const ASX::size_type s_contSize = (t_id < ID_one) ? 1 : t_id;
            typedef T_Element ASX_ASA[t_groupSize * s_contSize];

            static const ASX::size_type s_groupSize = t_groupSize;
            static const ASX::size_type s_groupNum = (t_maxSize + t_groupSize - 1) / t_groupSize;
            static const ASX::size_type s_maxSize = s_groupNum * s_groupSize;

            typedef T_Element ValueType;
            typedef T_Element AccessType;
            ASX_Array_common_types

                __host__ __device__ const_reference operator[] (difference_type index) const { return arr[index * s_contSize - (s_contSize - 1) * (index % s_groupSize)]; }
            __host__ __device__       reference operator[] (difference_type index) { return arr[index * s_contSize - (s_contSize - 1) * (index % s_groupSize)]; }
            __host__ __device__ const void* data()                      const { return arr; }
            __host__ __device__       void* data() { return arr; }
            ASX_Array_common_functions

                ValueType arr[s_maxSize * s_contSize];
        };

    };

    //======================================================
} // namespace
//======================================================

//------------------------------------------------------
// undef local defines
//------------------------------------------------------

#undef ASX_Array_common_types
#undef ASX_Array_common_functions
#undef ASX_Vector_common_functions

//------------------------------------------------------
// undef CUDA qualifiers
//------------------------------------------------------

#if !defined(__CUDACC__)
#undef __host__
#undef __device__
#endif

//------------------------------------------------------
#endif
