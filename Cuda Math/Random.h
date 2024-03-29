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

#ifndef _CML_RANDOM_
#define _CML_RANDOM_

#include "cuda_common.h"
#include "gl_common.h"

namespace cml
{
	CML_FUNC_DECL uint32_t lowerbias32(uint32_t x);
	CML_FUNC_DECL uint32_t rand();

	CML_FUNC_DECL float rand_float();
	CML_FUNC_DECL float rand_float(float min, float max);

	CML_FUNC_DECL int rand_int();
	CML_FUNC_DECL int rand_int(int min, int max);
}

#endif // _CML_RANDOM_