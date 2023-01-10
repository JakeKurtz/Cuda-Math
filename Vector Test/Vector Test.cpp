#include "pch.h"
#include "CppUnitTest.h"

#define _UNIT_TEST_

#include <Vector.h>

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace cml;

float nearest(float val, float step)
{
	return round(val / step);
}

namespace VectorTest
{
	TEST_CLASS(VectorTest)
	{
	public:
		TEST_METHOD(TestConstructor_Default)
		{
			vec4f v0 = vec4f();	Assert::IsTrue(v0.x == 0.f && v0.y == 0.f && v0.z == 0.f && v0.w == 0.f);
			vec3f v1 = vec3f();	Assert::IsTrue(v1.x == 0.f && v1.y == 0.f && v1.z == 0.f);
			vec2f v2 = vec2f();	Assert::IsTrue(v2.x == 0.f && v2.y == 0.f);
		}
		TEST_METHOD(TestConstructor_Regular)
		{
			vec4f v0 = vec4f(1, 2, 3, 4); Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			vec3f v1 = vec3f(1, 2, 3);	  Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);
			vec2f v2 = vec2f(1, 2);		  Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
		}
		TEST_METHOD(TestConstructor_Single_Values)
		{
			vec4f v0 = vec4f(3.1415926f); Assert::IsTrue(v0.x == 3.1415926f && v0.y == 3.1415926f && v0.z == 3.1415926f && v0.w == 3.1415926f);
			vec3f v1 = vec3f(3.1415926);  Assert::IsTrue(v1.x == 3.1415926f && v1.y == 3.1415926f && v1.z == 3.1415926f);
			vec2f v2 = vec2f(3.1415926);  Assert::IsTrue(v2.x == 3.1415926f && v2.y == 3.1415926f);
		}
		TEST_METHOD(TestConstructor_Vec4_Partial_Values_1)
		{
			vec4f v0;
			vec3f v1;
			vec2f v2;

			v1 = vec3f(1, 2, 3);
			v2 = vec2f(9, 8);

			v0 = vec4f(v1, 4);		Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			v0 = vec4f(v1);			Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 0.f);

			v0 = vec4f(v2, 7, 6);	Assert::IsTrue(v0.x == 9.f && v0.y == 8.f && v0.z == 7.f && v0.w == 6.f);
			v0 = vec4f(v2);			Assert::IsTrue(v0.x == 9.f && v0.y == 8.f && v0.z == 0.f && v0.w == 0.f);
		}
		TEST_METHOD(TestConstructor_Vec3_Partial_Values_1)
		{
			vec4f v0;
			vec3f v1;
			vec2f v2;

			v0 = vec4f(1, 2, 3, 4);
			v2 = vec2f(9, 8);

			v1 = vec3f(v0);
			Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);

			v1 = vec3f(v2, 7);
			Assert::IsTrue(v1.x == 9.f && v1.y == 8.f && v1.z == 7.f);

			v1 = vec3f(v2);
			Assert::IsTrue(v1.x == 9.f && v1.y == 8.f && v1.z == 0.f);
		}
		TEST_METHOD(TestConstructor_Vec2_Partial_Values_1)
		{
			vec4f v0;
			vec3f v1;
			vec2f v2;

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec3f(9, 8, 7);

			v2 = vec2f(v0);	Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
			v2 = vec2f(v1);	Assert::IsTrue(v2.x == 9.f && v2.y == 8.f);
		}
		TEST_METHOD(TestConstructor_Vec4_Partial_Values_2)
		{
			glm::vec4 a = glm::vec4(1.f, 2.f, 3.f, 4.f);
			glm::vec3 b = glm::vec3(5.f, 6.f, 7.f);
			glm::vec2 c = glm::vec2(8.f, 9.f);

			vec4f v;

			v = vec4f(a); Assert::IsTrue(v.x == 1.f && v.y == 2.f && v.z == 3.f && v.w == 4.f);
			v = vec4f(b); Assert::IsTrue(v.x == 5.f && v.y == 6.f && v.z == 7.f && v.w == 0.f);
			v = vec4f(c); Assert::IsTrue(v.x == 8.f && v.y == 9.f && v.z == 0.f && v.w == 0.f);
		}
		TEST_METHOD(TestConstructor_Vec3_Partial_Values_2)
		{
			glm::vec4 a = glm::vec4(1.f, 2.f, 3.f, 4.f);
			glm::vec3 b = glm::vec3(5.f, 6.f, 7.f);
			glm::vec2 c = glm::vec2(8.f, 9.f);

			vec3f v;

			v = vec3f(a); Assert::IsTrue(v.x == 1.f && v.y == 2.f && v.z == 3.f);
			v = vec3f(b); Assert::IsTrue(v.x == 5.f && v.y == 6.f && v.z == 7.f);
			v = vec3f(c); Assert::IsTrue(v.x == 8.f && v.y == 9.f && v.z == 0.f);
		}
		TEST_METHOD(TestConstructor_Vec2_Partial_Values_2)
		{
			glm::vec4 a = glm::vec4(1.f, 2.f, 3.f, 4.f);
			glm::vec3 b = glm::vec3(5.f, 6.f, 7.f);
			glm::vec2 c = glm::vec2(8.f, 9.f);

			vec2f v;

			v = vec2f(a); Assert::IsTrue(v.x == 1.f && v.y == 2.f);
			v = vec2f(b); Assert::IsTrue(v.x == 5.f && v.y == 6.f);
			v = vec2f(c); Assert::IsTrue(v.x == 8.f && v.y == 9.f);
		}

		TEST_METHOD(TestEqual_IsTrue)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(1, 2, 3, 4);

			Assert::IsTrue(x0 == y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(1, 2, 3);

			Assert::IsTrue(x1 == y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(1, 2);

			Assert::IsTrue(x2 == y2);
		}
		TEST_METHOD(TestEqual_IsFalse)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(4, 3, 2, 1);

			Assert::IsFalse(x0 == y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(7, 8, 9);

			Assert::IsFalse(x1 == y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(2, 1);

			Assert::IsFalse(x2 == y2);
		}

		TEST_METHOD(TestNotEqual_IsFalse)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(1, 2, 3, 4);

			Assert::IsFalse(x0 != y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(1, 2, 3);

			Assert::IsFalse(x1 != y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(1, 2);

			Assert::IsFalse(x2 != y2);
		}
		TEST_METHOD(TestNotEqual_IsTrue)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(4, 3, 2, 1);

			Assert::IsTrue(x0 != y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(7, 8, 9);

			Assert::IsTrue(x1 != y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(2, 1);

			Assert::IsTrue(x2 != y2);
		}

		TEST_METHOD(TestLessThan_IsTrue)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(2, 3, 4, 5);

			Assert::IsTrue(x0 < y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(2, 3, 4);

			Assert::IsTrue(x1 < y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(2, 3);

			Assert::IsTrue(x2 < y2);
		}
		TEST_METHOD(TestLessThan_IsFalse)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(0, 1, 2, 3);

			Assert::IsFalse(x0 < y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(0, 1, 2);

			Assert::IsFalse(x1 < y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(0, 1);

			Assert::IsFalse(x2 < y2);
		}

		TEST_METHOD(TestLEQ_IsTrue)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(2, 3, 4, 5);
			Assert::IsTrue(x0 <= y0);

			x0 = vec4f(1, 2, 3, 4);
			y0 = vec4f(1, 2, 3, 4);
			Assert::IsTrue(x0 <= y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(2, 3, 4);
			Assert::IsTrue(x1 <= y1);

			x1 = vec3f(1, 2, 3);
			y1 = vec3f(1, 2, 3);
			Assert::IsTrue(x1 <= y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(2, 3);
			Assert::IsTrue(x2 <= y2);

			x2 = vec2f(1, 2);
			y2 = vec2f(1, 2);
			Assert::IsTrue(x2 <= y2);
		}
		TEST_METHOD(TestLEQ_IsFalse)
		{
			vec4f y0 = vec4f(1, 2, 3, 4);
			vec4f x0 = vec4f(2, 3, 4, 5);
			Assert::IsFalse(x0 <= y0);

			vec3f y1 = vec3f(1, 2, 3);
			vec3f x1 = vec3f(2, 3, 4);
			Assert::IsFalse(x1 <= y1);

			vec2f y2 = vec2f(1, 2);
			vec2f x2 = vec2f(2, 3);
			Assert::IsFalse(x2 <= y2);
		}

		TEST_METHOD(TestGreaterThan_IsTrue)
		{
			vec4f y0 = vec4f(1, 2, 3, 4);
			vec4f x0 = vec4f(2, 3, 4, 5);

			Assert::IsTrue(x0 > y0);

			vec3f y1 = vec3f(1, 2, 3);
			vec3f x1 = vec3f(2, 3, 4);

			Assert::IsTrue(x1 > y1);

			vec2f y2 = vec2f(1, 2);
			vec2f x2 = vec2f(2, 3);

			Assert::IsTrue(x2 > y2);
		}
		TEST_METHOD(TestGreaterThan_IsFalse)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(2, 3, 4, 5);

			Assert::IsFalse(x0 > y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(2, 3, 4);

			Assert::IsFalse(x1 > y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(2, 3);

			Assert::IsFalse(x2 > y2);
		}

		TEST_METHOD(TestGEQ_IsTrue)
		{
			vec4f y0 = vec4f(1, 2, 3, 4);
			vec4f x0 = vec4f(2, 3, 4, 5);
			Assert::IsTrue(x0 >= y0);

			x0 = vec4f(1, 2, 3, 4);
			y0 = vec4f(1, 2, 3, 4);
			Assert::IsTrue(x0 >= y0);

			vec3f y1 = vec3f(1, 2, 3);
			vec3f x1 = vec3f(2, 3, 4);
			Assert::IsTrue(x1 >= y1);

			x1 = vec3f(1, 2, 3);
			y1 = vec3f(1, 2, 3);
			Assert::IsTrue(x1 >= y1);

			vec2f y2 = vec2f(1, 2);
			vec2f x2 = vec2f(2, 3);
			Assert::IsTrue(x2 >= y2);

			x2 = vec2f(1, 2);
			y2 = vec2f(1, 2);
			Assert::IsTrue(x2 >= y2);
		}
		TEST_METHOD(TestGEQ_IsFalse)
		{
			vec4f x0 = vec4f(1, 2, 3, 4);
			vec4f y0 = vec4f(2, 3, 4, 5);
			Assert::IsFalse(x0 >= y0);

			vec3f x1 = vec3f(1, 2, 3);
			vec3f y1 = vec3f(2, 3, 4);
			Assert::IsFalse(x1 >= y1);

			vec2f x2 = vec2f(1, 2);
			vec2f y2 = vec2f(2, 3);
			Assert::IsFalse(x2 >= y2);
		}

		TEST_METHOD(TestMul_Vec4)
		{
			vec4f v0, v1, v2;
			float s;

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec4f(0.1, 1123, 3.1415, 25);
			v2 = v0 * v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2246.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 9.4245f) &&
				trunc(1000. * v2.w) == trunc(1000. * 100.f));

			v0 = vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 * s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 2.f));

			v0 = vec4f(9, 8, 7, 6);
			s = 0.3333;
			v2 = s * v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.9997f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.6664f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.3331f) &&
				trunc(1000. * v2.w) == trunc(1000. * 1.9998f));

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec4f(5, 6, 7, 8);
			v1 *= v0;
			Assert::IsTrue(v1 == vec4f(5, 12, 21, 32));
		}
		TEST_METHOD(TestMul_Vec3)
		{
			vec3f v0, v1, v2;
			float s;

			v0 = vec3f(1, 2, 3);
			v1 = vec3f(0.1, 1123, 3.1415);
			v2 = v0 * v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2246.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 9.4245f));

			v0 = vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 * s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 1.5f));

			v0 = vec3f(9, 8, 7);
			s = 0.3333;
			v2 = s * v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.9997f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.6664f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.3331f));

			v0 = vec3f(1, 2, 3);
			v1 = vec3f(5, 6, 7);
			v1 *= v0;
			Assert::IsTrue(v1 == vec3f(5, 12, 21));
		}
		TEST_METHOD(TestMul_Vec2)
		{
			vec2f v0, v1, v2;
			float s;

			v0 = vec2f(1, 2);
			v1 = vec2f(0.1, 1123);
			v2 = v0 * v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2246.f));

			v0 = vec2f(1, 2);
			s = 0.5;
			v2 = v0 * s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.f));

			v0 = vec2f(9, 8);
			s = 0.3333;
			v2 = s * v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.9997f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.6664f));

			v0 = vec2f(1, 2);
			v1 = vec2f(5, 6);
			v1 *= v0;
			Assert::IsTrue(v1 == vec2f(5, 12));
		}

		TEST_METHOD(TestDiv_Vec4)
		{
			vec4f v0, v1, v2;
			float s;

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec4f(10, 34, 65, 89);
			v2 = v0 / v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 0.0588f) &&
				trunc(1000. * v2.z) == trunc(1000. * 0.0461f) &&
				trunc(1000. * v2.w) == trunc(1000. * 0.0449f));

			v0 = vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 / s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.w) == trunc(1000. * 8.f));

			v0 = vec4f(0.0123, 0.01388, 0.0158, 0.0185);
			s = 0.3333;
			v2 = s / v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 27.0975f) &&
				trunc(1000. * v2.y) == trunc(1000. * 24.0129f) &&
				trunc(1000. * v2.z) == trunc(1000. * 21.0949f) &&
				trunc(1000. * v2.w) == trunc(1000. * 18.0162f));

		}
		TEST_METHOD(TestDiv_Vec3)
		{
			vec3f v0, v1, v2;
			float s;

			v0 = vec3f(1, 2, 3);
			v1 = vec3f(10, 34, 65);
			v2 = v0 / v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 0.0588f) &&
				trunc(1000. * v2.z) == trunc(1000. * 0.0461f));

			v0 = vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 / s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 6.f));

			v0 = vec3f(0.0123, 0.01388, 0.0158);
			s = 0.3333;
			v2 = s / v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 27.0975f) &&
				trunc(1000. * v2.y) == trunc(1000. * 24.0129f) &&
				trunc(1000. * v2.z) == trunc(1000. * 21.0949f));

		}
		TEST_METHOD(TestDiv_Vec2)
		{
			vec2f v0, v1, v2;
			float s;

			v0 = vec2f(1, 2);
			v1 = vec2f(10, 34);
			v2 = v0 / v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 0.0588f));

			v0 = vec2f(1, 2);
			s = 0.5;
			v2 = v0 / s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 4.f));

			v0 = vec2f(0.0123, 0.01388);
			s = 0.3333;
			v2 = s / v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 27.0975f) &&
				trunc(1000. * v2.y) == trunc(1000. * 24.0129f));

		}

		TEST_METHOD(TestAdd_Vec4)
		{
			vec4f v0, v1, v2;
			float s;

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec4f(5, 6, 7, 8);
			v2 = v0 + v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 8.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 10.f) &&
				trunc(1000. * v2.w) == trunc(1000. * 12.f));

			v0 = vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 + s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 3.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 4.5f));

			v0 = vec4f(1, 2, 3, 4);
			s = -0.5;
			v2 = s + v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 3.5f));

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec4f(5, 6, 7, 8);
			v1 += v0;
			Assert::IsTrue(v1 == vec4f(6, 8, 10, 12));
		}
		TEST_METHOD(TestAdd_Vec3)
		{
			vec3f v0, v1, v2;
			float s;

			v0 = vec3f(1, 2, 3);
			v1 = vec3f(5, 6, 7);
			v2 = v0 + v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 8.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 10.f));

			v0 = vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 + s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 3.5f));

			v0 = vec3f(1, 2, 3);
			s = -0.5;
			v2 = s + v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f));

			v0 = vec3f(1, 2, 3);
			v1 = vec3f(5, 6, 7);
			v1 += v0;
			Assert::IsTrue(v1 == vec3f(6, 8, 10));
		}
		TEST_METHOD(TestAdd_Vec2)
		{
			vec2f v0, v1, v2;
			float s;

			v0 = vec2f(1, 2);
			v1 = vec2f(5, 6);
			v2 = v0 + v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 8.f));

			v0 = vec2f(1, 2);
			s = 0.5;
			v2 = v0 + s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.5f));

			v0 = vec2f(1, 2);
			s = -0.5;
			v2 = s + v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f));

			v0 = vec2f(1, 2);
			v1 = vec2f(5, 6);
			v1 += v0;
			Assert::IsTrue(v1 == vec2f(6, 8));
		}

		TEST_METHOD(TestSub_Vec4)
		{
			vec4f v0, v1, v2;
			float s;

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec4f(5, 6, 7, 8);
			v2 = v0 - v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.y) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.w) == trunc(1000. * -4.f));

			v0 = vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 - s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 3.5f));

			v0 = vec4f(1, 2, 3, 4);
			s = -0.5;
			v2 = s - v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * -2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * -3.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * -4.5f));
		}
		TEST_METHOD(TestSub_Vec3)
		{
			vec3f v0, v1, v2;
			float s;

			v0 = vec3f(1, 2, 3);
			v1 = vec3f(5, 6, 7);
			v2 = v0 - v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.y) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * -4.f));

			v0 = vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 - s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f));

			v0 = vec3f(1, 2, 3);
			s = -0.5;
			v2 = s - v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * -2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * -3.5f));
		}
		TEST_METHOD(TestSub_Vec2)
		{
			vec2f v0, v1, v2;
			float s;

			v0 = vec2f(1, 2);
			v1 = vec2f(5, 6);
			v2 = v0 - v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.y) == trunc(1000. * -4.f));

			v0 = vec2f(1, 2);
			s = 0.5;
			v2 = v0 - s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f));

			v0 = vec2f(1, 2);
			s = -0.5;
			v2 = s - v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * -2.5f));
		}

		TEST_METHOD(TestDot_Vec4)
		{
			vec4f v0, v1;
			float out;

			v0 = vec4f(1, 2, 3, 4);
			v1 = vec4f(5, 6, 7, 8);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 70.f));

			v0 = vec4f(0.182574, 0.365148, 0.547723, 0.730297);
			v1 = vec4f(0.379049, 0.454859, 0.530669, 0.606478);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.968864f));

			v0 = vec4f(1, 0, 0, 0);
			v1 = vec4f(1, 0, 0, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = vec4f(1, 0, 0, 0);
			v1 = vec4f(0, 0, 1, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));
		}
		TEST_METHOD(TestDot_Vec3)
		{
			vec3f v0, v1;
			float out;

			v0 = vec3f(1, 2, 3);
			v1 = vec3f(5, 6, 7);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 38.f));

			v0 = vec3f(0.182574, 0.365148, 0.547723);
			v1 = vec3f(0.379049, 0.454859, 0.530669);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.525955f));

			v0 = vec3f(1, 0, 0);
			v1 = vec3f(1, 0, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = vec3f(1, 0, 0);
			v1 = vec3f(0, 0, 1);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));
		}
		TEST_METHOD(TestDot_Vec2)
		{
			vec2f v0, v1;
			float out;

			v0 = vec2f(1, 2);
			v1 = vec2f(5, 6);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 17.f));

			v0 = vec2f(0.182574, 0.365148);
			v1 = vec2f(0.379049, 0.454859);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.235295f));

			v0 = vec2f(1, 0);
			v1 = vec2f(1, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = vec2f(1, 0);
			v1 = vec2f(0, 1);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));
		}

		TEST_METHOD(TestNegat)
		{
			vec4f v0 = vec4f(1, 2, 3, 4);			v0 = -v0;			Assert::IsTrue(v0.x == -1.f && v0.y == -2.f && v0.z == -3.f && v0.w == -4.f);
			vec3f v1 = vec3f(1, 2, 3);				v1 = -v1;			Assert::IsTrue(v1.x == -1.f && v1.y == -2.f && v1.z == -3.f);
			vec2f v2 = vec2f(1, 2);					v2 = -v2;			Assert::IsTrue(v2.x == -1.f && v2.y == -2.f);
		}

		TEST_METHOD(TestFloor)
		{
			vec4f v0 = vec4f(1.9, 2.9, 3.9, 4.9);	v0 = floor(v0);		Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			vec3f v1 = vec3f(1.9, 2.9, 3.9);		v1 = floor(v1);		Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);
			vec2f v2 = vec2f(1.9, 2.9);				v2 = floor(v2);		Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
		}

		TEST_METHOD(TestCeil)
		{
			vec4f v0 = vec4f(1.1, 2.1, 3.1, 4.1);	v0 = ceil(v0);		Assert::IsTrue(v0.x == 2.f && v0.y == 3.f && v0.z == 4.f && v0.w == 5.f);
			vec3f v1 = vec3f(1.1, 2.1, 3.1);		v1 = ceil(v1);		Assert::IsTrue(v1.x == 2.f && v1.y == 3.f && v1.z == 4.f);
			vec2f v2 = vec2f(1.1, 2.1);				v2 = ceil(v2);		Assert::IsTrue(v2.x == 2.f && v2.y == 3.f);
		}

		TEST_METHOD(TestFrac)
		{
			vec4f v0 = vec4f(1.1, 2.2, 3.3, 4.4);	v0 = frac(v0);

			Assert::IsTrue(
				trunc(::round(1000. * v0.x)) == trunc(1000. * 0.1) &&
				trunc(::round(1000. * v0.y)) == trunc(1000. * 0.2) &&
				trunc(::round(1000. * v0.z)) == trunc(1000. * 0.3) &&
				trunc(::round(1000. * v0.w)) == trunc(1000. * 0.4));

			vec3f v1 = vec3f(1.5, 2.6, 3.7);		v1 = frac(v1);

			Assert::IsTrue(
				trunc(::round(1000. * v1.x)) == trunc(1000. * 0.5) &&
				trunc(::round(1000. * v1.y)) == trunc(1000. * 0.6) &&
				trunc(::round(1000. * v1.z)) == trunc(1000. * 0.7));

			vec2f v2 = vec2f(1.8, 2.9);				v2 = frac(v2);

			Assert::IsTrue(
				trunc(::round(1000. * v2.x)) == trunc(1000. * 0.8) &&
				trunc(::round(1000. * v2.y)) == trunc(1000. * 0.9));
		}

		TEST_METHOD(TestAbs)
		{
			vec4f v0 = vec4f(-1, -2, -3, -4);		v0 = abs(v0);		Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			vec3f v1 = vec3f(-1, -2, -3);			v1 = abs(v1);		Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);
			vec2f v2 = vec2f(-1, -2);				v2 = abs(v2);		Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
		}

		TEST_METHOD(TestClamp_Vec4)
		{
			vec4f min_v = vec4f(0.f);
			vec4f max_v = vec4f(1.f);

			float min_f = -1.f;
			float max_f = 0.5f;

			vec4f v0 = vec4f(-10, 10, 0.5, 0.75);

			vec4f out = clamp(v0, min_v, max_v);	Assert::IsTrue(out.x == 0.f && out.y == 1.f && out.z == 0.5f && out.w == 0.75f);
			out = clamp(v0, min_f, max_f);			Assert::IsTrue(out.x == -1.f && out.y == 0.5f && out.z == 0.5f && out.w == 0.5f);
		}
		TEST_METHOD(TestClamp_Vec3)
		{
			vec3f min_v = vec3f(0.f);
			vec3f max_v = vec3f(1.f);

			float min_f = -1.f;
			float max_f = 0.5f;

			vec3f v0 = vec3f(-10, 0.5, 0.75);

			vec3f out = clamp(v0, min_v, max_v);	Assert::IsTrue(out.x == 0.f && out.y == 0.5f && out.z == 0.75f);
			out = clamp(v0, min_f, max_f);			Assert::IsTrue(out.x == -1.f && out.y == 0.5f && out.z == 0.5f);
		}
		TEST_METHOD(TestClamp_Vec2)
		{
			vec2f min_v = vec2f(0.f);
			vec2f max_v = vec2f(1.f);

			float min_f = -1.f;
			float max_f = 0.5f;

			vec2f v0 = vec2f(-10, 0.5);

			vec2f out = clamp(v0, min_v, max_v);	Assert::IsTrue(out.x == 0.f && out.y == 0.5f);
			out = clamp(v0, min_f, max_f);			Assert::IsTrue(out.x == -1.f && out.y == 0.5f);
		}

		TEST_METHOD(TestMax_Vec4)
		{
			vec4f v0, v1, v2;
			float s;

			v0 = vec4f(-1, 0, 1, -1);
			v1 = vec4f(-1, 1, 0, 0);

			v2 = max(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 1.f && v2.z == 1.f && v2.w == 0.f);

			s = 22.f;
			v2 = max(v0, s);
			Assert::IsTrue(v2.x == 22.f && v2.y == 22.f && v2.z == 22.f && v2.w == 22.f);

			s = -2.f;
			v2 = max(s, v0);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f && v2.w == -1.f);
		}
		TEST_METHOD(TestMax_Vec3)
		{
			vec3f v0, v1, v2;
			float s;

			v0 = vec3f(-1, 0, 1);
			v1 = vec3f(-1, 1, 0);

			v2 = max(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 1.f && v2.z == 1.f);

			s = 22.f;
			v2 = max(v0, s);
			Assert::IsTrue(v2.x == 22.f && v2.y == 22.f && v2.z == 22.f);

			s = -2.f;
			v2 = max(s, v0);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f);
		}
		TEST_METHOD(TestMax_Vec2)
		{
			vec2f v0, v1, v2;
			float s;

			v0 = vec2f(-1, 0);
			v1 = vec2f(-1, 1);

			v2 = max(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 1.f);

			s = 22.f;
			v2 = max(v0, s);
			Assert::IsTrue(v2.x == 22.f && v2.y == 22.f);

			s = -2.f;
			v2 = max(s, v0);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f);
		}

		TEST_METHOD(TestMin_Vec4)
		{
			vec4f v0, v1, v2;
			float s;

			v0 = vec4f(-1, 0, 1, -1);
			v1 = vec4f(-1, 1, 0, 0);

			v2 = min(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 0.f && v2.w == -1.f);

			s = 22.f;
			v2 = min(v0, s);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f && v2.w == -1.f);

			s = -2.f;
			v2 = min(s, v0);
			Assert::IsTrue(v2.x == -2.f && v2.y == -2.f && v2.z == -2.f && v2.w == -2.f);
		}
		TEST_METHOD(TestMin_Vec3)
		{
			vec3f v0, v1, v2;
			float s;

			v0 = vec3f(-1, 0, 1);
			v1 = vec3f(-1, 1, 0);

			v2 = min(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 0.f);

			s = 22.f;
			v2 = min(v0, s);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f);

			s = -2.f;
			v2 = min(s, v0);
			Assert::IsTrue(v2.x == -2.f && v2.y == -2.f && v2.z == -2.f);
		}
		TEST_METHOD(TestMin_Vec2)
		{
			vec2f v0, v1, v2;
			float s;

			v0 = vec2f(-1, 0);
			v1 = vec2f(-1, 1);

			v2 = min(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f);

			s = 22.f;
			v2 = min(v0, s);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f);

			s = -2.f;
			v2 = min(s, v0);
			Assert::IsTrue(v2.x == -2.f && v2.y == -2.f);
		}

		TEST_METHOD(TestPow)
		{
			vec4f v0; v0 = vec4f(1, 2, 3, 4); v0 = pow(v0, 2.f); Assert::IsTrue(v0.x == 1.f && v0.y == 4.f && v0.z == 9.f && v0.w == 16.f);
			vec3f v1; v1 = vec3f(1, 2, 3); v1 = pow(v1, 2.f); Assert::IsTrue(v1.x == 1.f && v1.y == 4.f && v1.z == 9.f);
			vec2f v2; v2 = vec2f(1, 2); v2 = pow(v2, 2.f); Assert::IsTrue(v2.x == 1.f && v2.y == 4.f);
		}

		TEST_METHOD(TestLength_Vec4)
		{
			vec4f v0;
			float out;

			v0 = vec4f(0);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));

			v0 = vec4f(0.182574, 0.365148, 0.547723, 0.730297);
			out = 0.001f * round(length(v0) / 0.001f);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = vec4f(1, 2, 3, 4);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 5.47723f));
		}
		TEST_METHOD(TestLength_Vec3)
		{
			vec3f v0;
			float out;

			v0 = vec3f(0);
			out = nearest(length(v0), 0.001f);
			Assert::IsTrue(out == 0.f);

			v0 = vec3f(0.267261, 0.534522, 0.801783);
			out = nearest(length(v0), 0.001f);
			Assert::IsTrue(out == 1000.f);

			v0 = vec3f(1, 2, 3);
			out = nearest(length(v0), 0.001f);
			Assert::IsTrue(out == 3742.f);
		}
		TEST_METHOD(TestLength_Vec2)
		{
			vec2f v0;
			float out;

			v0 = vec2f(0);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));

			v0 = vec2f(0.447213, 0.894427);
			out = 0.001f * round(length(v0) / 0.001f);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = vec2f(1, 2);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 2.23606f));
		}

		TEST_METHOD(TestDistance_Vec4)
		{
			vec4f v0, v1;
			float d;

			v0 = vec4f(1, 2, 3, 4); v1 = vec4f(5, 6, 7, 8); d = distance(v0, v1); Assert::IsTrue(d == 8.f);
			v0 = vec4f(1, 2, 3, 4); v1 = vec4f(1, 2, 3, 4); d = distance(v0, v1); Assert::IsTrue(d == 0.f);
			v0 = vec4f(1, 2, 3, 4); v1 = vec4f(-6, -7, -8, -9); d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 204939.f);
		}
		TEST_METHOD(TestDistance_Vec3)
		{
			vec3f v0, v1;
			float d;

			v0 = vec3f(1, 2, 3); v1 = vec3f(5, 6, 7);	 d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 69282.f);
			v0 = vec3f(1, 2, 3); v1 = vec3f(1, 2, 3);	 d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 0.f);
			v0 = vec3f(1, 2, 3); v1 = vec3f(-6, -7, -8); d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 158430.f);
		}
		TEST_METHOD(TestDistance_Vec2)
		{
			vec2f v0, v1;
			float d;

			v0 = vec2f(2, 3); v1 = vec2f(5, 6);   d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 42426.f);
			v0 = vec2f(2, 3); v1 = vec2f(2, 3);	  d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 0.f);
			v0 = vec2f(2, 3); v1 = vec2f(-6, -7); d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 128062.f);
		}

		TEST_METHOD(TestNormalize_Vec4)
		{
			Assert::IsTrue(::round(length(normalize(vec4d(1, 2, 3, 4)))) == 1.f);
			Assert::IsTrue(::round(length(normalize(vec4d(2)))) == 1.f);
			Assert::IsTrue(normalize(vec4f(2)) == vec4f(0.5));
			Assert::IsTrue(normalize(vec4f(0.f)) == vec4f(0.f));

			vec4f v = normalize(vec4f(1, 2, 3, 4));

			Assert::IsTrue(
				trunc(1000. * v.x) == trunc(1000. * 0.1825f) &&
				trunc(1000. * v.y) == trunc(1000. * 0.3651f) &&
				trunc(1000. * v.z) == trunc(1000. * 0.5477f) &&
				trunc(1000. * v.w) == trunc(1000. * 0.7302f));
		}
		TEST_METHOD(TestNormalize_Vec3)
		{
			Assert::IsTrue(::round(length(normalize(vec3f(1, 2, 3)))) == 1.f);
			Assert::IsTrue(::round(length(normalize(vec3f(2)))) == 1.f);

			Assert::IsTrue(normalize(vec3f(0.f)) == vec3f(0.f));

			vec3f v = normalize(vec3f(1, 2, 3));

			Assert::IsTrue(
				trunc(1000. * v.x) == trunc(1000. * 0.2672f) &&
				trunc(1000. * v.y) == trunc(1000. * 0.5345f) &&
				trunc(1000. * v.z) == trunc(1000. * 0.8017f));

			v = normalize(vec3f(2));

			Assert::IsTrue(
				trunc(1000. * v.x) == trunc(1000. * 0.5773f) &&
				trunc(1000. * v.y) == trunc(1000. * 0.5773f) &&
				trunc(1000. * v.z) == trunc(1000. * 0.5773f));
		}
		TEST_METHOD(TestNormalize_Vec2)
		{
			Assert::IsTrue(::round(length(normalize(vec2f(4, 2)))) == 1.f);
			Assert::IsTrue(::round(length(normalize(vec2f(2)))) == 1.f);

			Assert::IsTrue(normalize(vec2f(0.f)) == vec2f(0.f));

			vec2f v = normalize(vec2f(4, 2));

			Assert::IsTrue(
				trunc(1000. * v.x) == trunc(1000. * 0.8944f) &&
				trunc(1000. * v.y) == trunc(1000. * 0.4472f));

			v = normalize(vec2f(2));

			Assert::IsTrue(
				trunc(1000. * v.x) == trunc(1000. * 0.7071f) &&
				trunc(1000. * v.y) == trunc(1000. * 0.7071f));
		}

		TEST_METHOD(TestReflect)
		{
			vec3f n(0, 1, 0), r, i;

			i = normalize(vec3f(1,2,3));
			r = reflect(i, n);

			Assert::IsTrue(i == abs(r));
			Assert::IsTrue(dot(i,n) == dot(abs(r),n));

			i = normalize(vec3f(42, 2, 3.1415));
			r = reflect(i, n);

			Assert::IsTrue(i == abs(r));
			Assert::IsTrue(dot(i, n) == dot(abs(r), n));
		}

		TEST_METHOD(TestCross)
		{
			Assert::IsTrue(cross(vec3f(1, 2, 3), vec3f(3, 4, 5)) == vec3f(-2, 4, -2));
			Assert::IsTrue(cross(vec3f(-1, 5, -20), vec3f(3, 4, 5)) == vec3f(105, -55, -19));
			Assert::IsTrue(cross(vec3f(0, 1, 0), vec3f(-1, 0, 0)) == vec3f(0, 0, 1));
			Assert::IsTrue(cross(vec3f(0), vec3f(0)) == vec3f(0));
			Assert::IsTrue(cross(vec3f(1), vec3f(1)) == vec3f(0));
		}

		//TEST_METHOD(TestMix)
		//}
		//	Assert::IsTrue(false);
		//}
	};
}
