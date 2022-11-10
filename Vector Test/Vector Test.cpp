#include "pch.h"
#include "CppUnitTest.h"

#define _TEST_

#include "../Cuda Math/Vector.h"

using namespace Microsoft::VisualStudio::CppUnitTestFramework;
using namespace jek;

float nearest(float val, float step)
{
	return step * round(val / step);
}

namespace VectorTest
{
	TEST_CLASS(VectorTest)
	{
	public:
		TEST_METHOD(TestConstructor_0)
		{
			Vec4f v0 = Vec4f();	Assert::IsTrue(v0.x == 0.f && v0.y == 0.f && v0.z == 0.f && v0.w == 0.f);
			Vec3f v1 = Vec3f();	Assert::IsTrue(v1.x == 0.f && v1.y == 0.f && v1.z == 0.f);
			Vec2f v2 = Vec2f();	Assert::IsTrue(v2.x == 0.f && v2.y == 0.f);
		}
		TEST_METHOD(TestConstructor_1)
		{
			Vec4f v0 = Vec4f(1, 2, 3, 4); Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			Vec3f v1 = Vec3f(1, 2, 3);	  Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);
			Vec2f v2 = Vec2f(1, 2);		  Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
		}
		TEST_METHOD(TestConstructor_2)
		{
			Vec4f v0 = Vec4f(3.1415926f); Assert::IsTrue(v0.x == 3.1415926f && v0.y == 3.1415926f && v0.z == 3.1415926f && v0.w == 3.1415926f);
			Vec3f v1 = Vec3f(3.1415926);  Assert::IsTrue(v1.x == 3.1415926f && v1.y == 3.1415926f && v1.z == 3.1415926f);
			Vec2f v2 = Vec2f(3.1415926);  Assert::IsTrue(v2.x == 3.1415926f && v2.y == 3.1415926f);
		}
		TEST_METHOD(TestConstructor_3)
		{
			Vec4f v0;
			Vec3f v1;
			Vec2f v2;

			v1 = Vec3f(1, 2, 3);
			v2 = Vec2f(9, 8);

			v0 = Vec4f(v1, 4);		Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			v0 = Vec4f(v1);			Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 0.f);

			v0 = Vec4f(v2, 7, 6);	Assert::IsTrue(v0.x == 9.f && v0.y == 8.f && v0.z == 7.f && v0.w == 6.f);
			v0 = Vec4f(v2);			Assert::IsTrue(v0.x == 9.f && v0.y == 8.f && v0.z == 0.f && v0.w == 0.f);
		}
		TEST_METHOD(TestConstructor_4)
		{
			Vec4f v0;
			Vec3f v1;
			Vec2f v2;

			v0 = Vec4f(1, 2, 3, 4);
			v2 = Vec2f(9, 8);

			v1 = Vec3f(v0);
			Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);

			v1 = Vec3f(v2, 7);
			Assert::IsTrue(v1.x == 9.f && v1.y == 8.f && v1.z == 7.f);

			v1 = Vec3f(v2);
			Assert::IsTrue(v1.x == 9.f && v1.y == 8.f && v1.z == 0.f);
		}
		TEST_METHOD(TestConstructor_5)
		{
			Vec4f v0;
			Vec3f v1;
			Vec2f v2;

			v0 = Vec4f(1, 2, 3, 4);
			v1 = Vec3f(9, 8, 7);

			v2 = Vec2f(v0);	Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
			v2 = Vec2f(v1);	Assert::IsTrue(v2.x == 9.f && v2.y == 8.f);
		}
		TEST_METHOD(TestConstructor_6)
		{
			glm::vec4 a = glm::vec4(1.f, 2.f, 3.f, 4.f);
			glm::vec3 b = glm::vec3(5.f, 6.f, 7.f);
			glm::vec2 c = glm::vec2(8.f, 9.f);

			Vec4f v;

			v = Vec4f(a); Assert::IsTrue(v.x == 1.f && v.y == 2.f && v.z == 3.f && v.w == 4.f);
			v = Vec4f(b); Assert::IsTrue(v.x == 5.f && v.y == 6.f && v.z == 7.f && v.w == 0.f);
			v = Vec4f(c); Assert::IsTrue(v.x == 8.f && v.y == 9.f && v.z == 0.f && v.w == 0.f);
		}
		TEST_METHOD(TestConstructor_7)
		{
			glm::vec4 a = glm::vec4(1.f, 2.f, 3.f, 4.f);
			glm::vec3 b = glm::vec3(5.f, 6.f, 7.f);
			glm::vec2 c = glm::vec2(8.f, 9.f);

			Vec3f v;

			v = Vec3f(a); Assert::IsTrue(v.x == 1.f && v.y == 2.f && v.z == 3.f);
			v = Vec3f(b); Assert::IsTrue(v.x == 5.f && v.y == 6.f && v.z == 7.f);
			v = Vec3f(c); Assert::IsTrue(v.x == 8.f && v.y == 9.f && v.z == 0.f);
		}
		TEST_METHOD(TestConstructor_8)
		{
			glm::vec4 a = glm::vec4(1.f, 2.f, 3.f, 4.f);
			glm::vec3 b = glm::vec3(5.f, 6.f, 7.f);
			glm::vec2 c = glm::vec2(8.f, 9.f);

			Vec2f v;

			v = Vec2f(a); Assert::IsTrue(v.x == 1.f && v.y == 2.f);
			v = Vec2f(b); Assert::IsTrue(v.x == 5.f && v.y == 6.f);
			v = Vec2f(c); Assert::IsTrue(v.x == 8.f && v.y == 9.f);
		}

		TEST_METHOD(TestEqual_0)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(1, 2, 3, 4);

			Assert::IsTrue(x0 == y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(1, 2, 3);

			Assert::IsTrue(x1 == y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(1, 2);

			Assert::IsTrue(x2 == y2);
		}
		TEST_METHOD(TestEqual_1)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(4, 3, 2, 1);

			Assert::IsFalse(x0 == y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(7, 8, 9);

			Assert::IsFalse(x1 == y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(2, 1);

			Assert::IsFalse(x2 == y2);
		}

		TEST_METHOD(TestNotEqual_0)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(1, 2, 3, 4);

			Assert::IsFalse(x0 != y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(1, 2, 3);

			Assert::IsFalse(x1 != y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(1, 2);

			Assert::IsFalse(x2 != y2);
		}
		TEST_METHOD(TestNotEqual_1)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(4, 3, 2, 1);

			Assert::IsTrue(x0 != y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(7, 8, 9);

			Assert::IsTrue(x1 != y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(2, 1);

			Assert::IsTrue(x2 != y2);
		}

		TEST_METHOD(TestLessThan_0)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(2, 3, 4, 5);

			Assert::IsTrue(x0 < y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(2, 3, 4);

			Assert::IsTrue(x1 < y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(2, 3);

			Assert::IsTrue(x2 < y2);
		}
		TEST_METHOD(TestLessThan_1)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(0, 1, 2, 3);

			Assert::IsFalse(x0 < y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(0, 1, 2);

			Assert::IsFalse(x1 < y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(0, 1);

			Assert::IsFalse(x2 < y2);
		}

		TEST_METHOD(TestLEQ_0)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(2, 3, 4, 5);
			Assert::IsTrue(x0 <= y0);

			x0 = Vec4f(1, 2, 3, 4);
			y0 = Vec4f(1, 2, 3, 4);
			Assert::IsTrue(x0 <= y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(2, 3, 4);
			Assert::IsTrue(x1 <= y1);

			x1 = Vec3f(1, 2, 3);
			y1 = Vec3f(1, 2, 3);
			Assert::IsTrue(x1 <= y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(2, 3);
			Assert::IsTrue(x2 <= y2);

			x2 = Vec2f(1, 2);
			y2 = Vec2f(1, 2);
			Assert::IsTrue(x2 <= y2);
		}
		TEST_METHOD(TestLEQ_1)
		{
			Vec4f y0 = Vec4f(1, 2, 3, 4);
			Vec4f x0 = Vec4f(2, 3, 4, 5);
			Assert::IsFalse(x0 <= y0);

			Vec3f y1 = Vec3f(1, 2, 3);
			Vec3f x1 = Vec3f(2, 3, 4);
			Assert::IsFalse(x1 <= y1);

			Vec2f y2 = Vec2f(1, 2);
			Vec2f x2 = Vec2f(2, 3);
			Assert::IsFalse(x2 <= y2);
		}

		TEST_METHOD(TestGreaterThan_0)
		{
			Vec4f y0 = Vec4f(1, 2, 3, 4);
			Vec4f x0 = Vec4f(2, 3, 4, 5);

			Assert::IsTrue(x0 > y0);

			Vec3f y1 = Vec3f(1, 2, 3);
			Vec3f x1 = Vec3f(2, 3, 4);

			Assert::IsTrue(x1 > y1);

			Vec2f y2 = Vec2f(1, 2);
			Vec2f x2 = Vec2f(2, 3);

			Assert::IsTrue(x2 > y2);
		}
		TEST_METHOD(TestGreaterThan_1)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(2, 3, 4, 5);

			Assert::IsFalse(x0 > y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(2, 3, 4);

			Assert::IsFalse(x1 > y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(2, 3);

			Assert::IsFalse(x2 > y2);
		}

		TEST_METHOD(TestGEQ_0)
		{
			Vec4f y0 = Vec4f(1, 2, 3, 4);
			Vec4f x0 = Vec4f(2, 3, 4, 5);
			Assert::IsTrue(x0 >= y0);

			x0 = Vec4f(1, 2, 3, 4);
			y0 = Vec4f(1, 2, 3, 4);
			Assert::IsTrue(x0 >= y0);

			Vec3f y1 = Vec3f(1, 2, 3);
			Vec3f x1 = Vec3f(2, 3, 4);
			Assert::IsTrue(x1 >= y1);

			x1 = Vec3f(1, 2, 3);
			y1 = Vec3f(1, 2, 3);
			Assert::IsTrue(x1 >= y1);

			Vec2f y2 = Vec2f(1, 2);
			Vec2f x2 = Vec2f(2, 3);
			Assert::IsTrue(x2 >= y2);

			x2 = Vec2f(1, 2);
			y2 = Vec2f(1, 2);
			Assert::IsTrue(x2 >= y2);
		}
		TEST_METHOD(TestGEQ_1)
		{
			Vec4f x0 = Vec4f(1, 2, 3, 4);
			Vec4f y0 = Vec4f(2, 3, 4, 5);
			Assert::IsFalse(x0 >= y0);

			Vec3f x1 = Vec3f(1, 2, 3);
			Vec3f y1 = Vec3f(2, 3, 4);
			Assert::IsFalse(x1 >= y1);

			Vec2f x2 = Vec2f(1, 2);
			Vec2f y2 = Vec2f(2, 3);
			Assert::IsFalse(x2 >= y2);
		}

		TEST_METHOD(TestMul_0)
		{
			Vec4f v0, v1, v2;
			float s;

			v0 = Vec4f(1, 2, 3, 4);
			v1 = Vec4f(0.1, 1123, 3.1415, 25);
			v2 = v0 * v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2246.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 9.4245f) &&
				trunc(1000. * v2.w) == trunc(1000. * 100.f));

			v0 = Vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 * s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 2.f));

			v0 = Vec4f(9, 8, 7, 6);
			s = 0.3333;
			v2 = s * v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.9997f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.6664f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.3331f) &&
				trunc(1000. * v2.w) == trunc(1000. * 1.9998f));
		}
		TEST_METHOD(TestMul_1)
		{
			Vec3f v0, v1, v2;
			float s;

			v0 = Vec3f(1, 2, 3);
			v1 = Vec3f(0.1, 1123, 3.1415);
			v2 = v0 * v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2246.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 9.4245f));

			v0 = Vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 * s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 1.5f));

			v0 = Vec3f(9, 8, 7);
			s = 0.3333;
			v2 = s * v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.9997f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.6664f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.3331f));
		}
		TEST_METHOD(TestMul_2)
		{
			Vec2f v0, v1, v2;
			float s;

			v0 = Vec2f(1, 2);
			v1 = Vec2f(0.1, 1123);
			v2 = v0 * v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2246.f));

			v0 = Vec2f(1, 2);
			s = 0.5;
			v2 = v0 * s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.f));

			v0 = Vec2f(9, 8);
			s = 0.3333;
			v2 = s * v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.9997f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.6664f));
		}

		TEST_METHOD(TestDiv_0)
		{
			Vec4f v0, v1, v2;
			float s;

			v0 = Vec4f(1, 2, 3, 4);
			v1 = Vec4f(10, 34, 65, 89);
			v2 = v0 / v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 0.0588f) &&
				trunc(1000. * v2.z) == trunc(1000. * 0.0461f) &&
				trunc(1000. * v2.w) == trunc(1000. * 0.0449f));

			v0 = Vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 / s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.w) == trunc(1000. * 8.f));

			v0 = Vec4f(0.0123, 0.01388, 0.0158, 0.0185);
			s = 0.3333;
			v2 = s / v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 27.0975f) &&
				trunc(1000. * v2.y) == trunc(1000. * 24.0129f) &&
				trunc(1000. * v2.z) == trunc(1000. * 21.0949f) &&
				trunc(1000. * v2.w) == trunc(1000. * 18.0162f));

		}
		TEST_METHOD(TestDiv_1)
		{
			Vec3f v0, v1, v2;
			float s;

			v0 = Vec3f(1, 2, 3);
			v1 = Vec3f(10, 34, 65);
			v2 = v0 / v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 0.0588f) &&
				trunc(1000. * v2.z) == trunc(1000. * 0.0461f));

			v0 = Vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 / s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 6.f));

			v0 = Vec3f(0.0123, 0.01388, 0.0158);
			s = 0.3333;
			v2 = s / v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 27.0975f) &&
				trunc(1000. * v2.y) == trunc(1000. * 24.0129f) &&
				trunc(1000. * v2.z) == trunc(1000. * 21.0949f));

		}
		TEST_METHOD(TestDiv_2)
		{
			Vec2f v0, v1, v2;
			float s;

			v0 = Vec2f(1, 2);
			v1 = Vec2f(10, 34);
			v2 = v0 / v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.1f) &&
				trunc(1000. * v2.y) == trunc(1000. * 0.0588f));

			v0 = Vec2f(1, 2);
			s = 0.5;
			v2 = v0 / s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 2.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 4.f));

			v0 = Vec2f(0.0123, 0.01388);
			s = 0.3333;
			v2 = s / v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 27.0975f) &&
				trunc(1000. * v2.y) == trunc(1000. * 24.0129f));

		}

		TEST_METHOD(TestAdd_0)
		{
			Vec4f v0, v1, v2;
			float s;

			v0 = Vec4f(1, 2, 3, 4);
			v1 = Vec4f(5, 6, 7, 8);
			v2 = v0 + v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 8.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 10.f) &&
				trunc(1000. * v2.w) == trunc(1000. * 12.f));

			v0 = Vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 + s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 3.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 4.5f));

			v0 = Vec4f(1, 2, 3, 4);
			s = -0.5;
			v2 = s + v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 3.5f));
		}
		TEST_METHOD(TestAdd_1)
		{
			Vec3f v0, v1, v2;
			float s;

			v0 = Vec3f(1, 2, 3);
			v1 = Vec3f(5, 6, 7);
			v2 = v0 + v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 8.f) &&
				trunc(1000. * v2.z) == trunc(1000. * 10.f));

			v0 = Vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 + s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 3.5f));

			v0 = Vec3f(1, 2, 3);
			s = -0.5;
			v2 = s + v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f));
		}
		TEST_METHOD(TestAdd_2)
		{
			Vec2f v0, v1, v2;
			float s;

			v0 = Vec2f(1, 2);
			v1 = Vec2f(5, 6);
			v2 = v0 + v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 6.f) &&
				trunc(1000. * v2.y) == trunc(1000. * 8.f));

			v0 = Vec2f(1, 2);
			s = 0.5;
			v2 = v0 + s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 2.5f));

			v0 = Vec2f(1, 2);
			s = -0.5;
			v2 = s + v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f));
		}

		TEST_METHOD(TestSub_0)
		{
			Vec4f v0, v1, v2;
			float s;

			v0 = Vec4f(1, 2, 3, 4);
			v1 = Vec4f(5, 6, 7, 8);
			v2 = v0 - v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.y) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.w) == trunc(1000. * -4.f));

			v0 = Vec4f(1, 2, 3, 4);
			s = 0.5;
			v2 = v0 - s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * 3.5f));

			v0 = Vec4f(1, 2, 3, 4);
			s = -0.5;
			v2 = s - v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * -2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * -3.5f) &&
				trunc(1000. * v2.w) == trunc(1000. * -4.5f));
		}
		TEST_METHOD(TestSub_1)
		{
			Vec3f v0, v1, v2;
			float s;

			v0 = Vec3f(1, 2, 3);
			v1 = Vec3f(5, 6, 7);
			v2 = v0 - v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.y) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.z) == trunc(1000. * -4.f));

			v0 = Vec3f(1, 2, 3);
			s = 0.5;
			v2 = v0 - s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * 2.5f));

			v0 = Vec3f(1, 2, 3);
			s = -0.5;
			v2 = s - v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * -2.5f) &&
				trunc(1000. * v2.z) == trunc(1000. * -3.5f));
		}
		TEST_METHOD(TestSub_2)
		{
			Vec2f v0, v1, v2;
			float s;

			v0 = Vec2f(1, 2);
			v1 = Vec2f(5, 6);
			v2 = v0 - v1;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -4.f) &&
				trunc(1000. * v2.y) == trunc(1000. * -4.f));

			v0 = Vec2f(1, 2);
			s = 0.5;
			v2 = v0 - s;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * 0.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * 1.5f));

			v0 = Vec2f(1, 2);
			s = -0.5;
			v2 = s - v0;
			Assert::IsTrue(
				trunc(1000. * v2.x) == trunc(1000. * -1.5f) &&
				trunc(1000. * v2.y) == trunc(1000. * -2.5f));
		}

		TEST_METHOD(TestDot_0)
		{
			Vec4f v0, v1;
			float out;

			v0 = Vec4f(1, 2, 3, 4);
			v1 = Vec4f(5, 6, 7, 8);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 70.f));

			v0 = Vec4f(0.182574, 0.365148, 0.547723, 0.730297);
			v1 = Vec4f(0.379049, 0.454859, 0.530669, 0.606478);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.968864f));

			v0 = Vec4f(1, 0, 0, 0);
			v1 = Vec4f(1, 0, 0, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = Vec4f(1, 0, 0, 0);
			v1 = Vec4f(0, 0, 1, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));
		}
		TEST_METHOD(TestDot_1)
		{
			Vec3f v0, v1;
			float out;

			v0 = Vec3f(1, 2, 3);
			v1 = Vec3f(5, 6, 7);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 38.f));

			v0 = Vec3f(0.182574, 0.365148, 0.547723);
			v1 = Vec3f(0.379049, 0.454859, 0.530669);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.525955f));

			v0 = Vec3f(1, 0, 0);
			v1 = Vec3f(1, 0, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = Vec3f(1, 0, 0);
			v1 = Vec3f(0, 0, 1);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));
		}
		TEST_METHOD(TestDot_2)
		{
			Vec2f v0, v1;
			float out;

			v0 = Vec2f(1, 2);
			v1 = Vec2f(5, 6);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 17.f));

			v0 = Vec2f(0.182574, 0.365148);
			v1 = Vec2f(0.379049, 0.454859);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.235295f));

			v0 = Vec2f(1, 0);
			v1 = Vec2f(1, 0);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = Vec2f(1, 0);
			v1 = Vec2f(0, 1);
			out = dot(v0, v1);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));
		}

		TEST_METHOD(TestNegat_0)
		{
			Vec4f v0 = Vec4f(1, 2, 3, 4);			v0 = -v0;			Assert::IsTrue(v0.x == -1.f && v0.y == -2.f && v0.z == -3.f && v0.w == -4.f);
			Vec3f v1 = Vec3f(1, 2, 3);				v1 = -v1;			Assert::IsTrue(v1.x == -1.f && v1.y == -2.f && v1.z == -3.f);
			Vec2f v2 = Vec2f(1, 2);					v2 = -v2;			Assert::IsTrue(v2.x == -1.f && v2.y == -2.f);
		}

		TEST_METHOD(TestFloor_0)
		{
			Vec4f v0 = Vec4f(1.9, 2.9, 3.9, 4.9);	v0 = floor(v0);		Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			Vec3f v1 = Vec3f(1.9, 2.9, 3.9);		v1 = floor(v1);		Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);
			Vec2f v2 = Vec2f(1.9, 2.9);				v2 = floor(v2);		Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
		}

		TEST_METHOD(TestCeil_0)
		{
			Vec4f v0 = Vec4f(1.1, 2.1, 3.1, 4.1);	v0 = ceil(v0);		Assert::IsTrue(v0.x == 2.f && v0.y == 3.f && v0.z == 4.f && v0.w == 5.f);
			Vec3f v1 = Vec3f(1.1, 2.1, 3.1);		v1 = ceil(v1);		Assert::IsTrue(v1.x == 2.f && v1.y == 3.f && v1.z == 4.f);
			Vec2f v2 = Vec2f(1.1, 2.1);				v2 = ceil(v2);		Assert::IsTrue(v2.x == 2.f && v2.y == 3.f);
		}

		TEST_METHOD(TestFrac_0)
		{
			Vec4f v0 = Vec4f(1.1, 2.2, 3.3, 4.4);	v0 = frac(v0);		Assert::IsTrue(v0.x == 0.1f && v0.y == 0.2f && v0.z == 0.3f && v0.w == 0.4f);
			Vec3f v1 = Vec3f(1.5, 2.6, 3.7);		v1 = frac(v1);		Assert::IsTrue(v1.x == 0.5f && v1.y == 0.6f && v1.z == 0.7f);
			Vec2f v2 = Vec2f(1.8, 2.9);				v2 = frac(v2);		Assert::IsTrue(v2.x == 0.8f && v2.y == 0.9f);
		}

		TEST_METHOD(TestAbs_0)
		{
			Vec4f v0 = Vec4f(-1, -2, -3, -4);		v0 = abs(v0);		Assert::IsTrue(v0.x == 1.f && v0.y == 2.f && v0.z == 3.f && v0.w == 4.f);
			Vec3f v1 = Vec3f(-1, -2, -3);			v1 = abs(v1);		Assert::IsTrue(v1.x == 1.f && v1.y == 2.f && v1.z == 3.f);
			Vec2f v2 = Vec2f(-1, -2);				v2 = abs(v2);		Assert::IsTrue(v2.x == 1.f && v2.y == 2.f);
		}

		TEST_METHOD(TestClamp_0)
		{
			Vec4f min_v = Vec4f(0.f);
			Vec4f max_v = Vec4f(1.f);

			float min_f = -1.f;
			float max_f = 0.5f;

			Vec4f v0 = Vec4f(-10, 10, 0.5, 0.75);

			Vec4f out = clamp(v0, min_v, max_v);	Assert::IsTrue(out.x == 0.f && out.y == 1.f && out.z == 0.5f && out.w == 0.75f);
			out = clamp(v0, min_f, max_f);			Assert::IsTrue(out.x == -1.f && out.y == 0.5f && out.z == 0.5f && out.w == 0.5f);
		}
		TEST_METHOD(TestClamp_1)
		{
			Vec3f min_v = Vec3f(0.f);
			Vec3f max_v = Vec3f(1.f);

			float min_f = -1.f;
			float max_f = 0.5f;

			Vec3f v0 = Vec3f(-10, 0.5, 0.75);

			Vec3f out = clamp(v0, min_v, max_v);	Assert::IsTrue(out.x == 0.f && out.y == 0.5f && out.z == 0.75f);
			out = clamp(v0, min_f, max_f);			Assert::IsTrue(out.x == -1.f && out.y == 0.5f && out.z == 0.5f);
		}
		TEST_METHOD(TestClamp_2)
		{
			Vec2f min_v = Vec2f(0.f);
			Vec2f max_v = Vec2f(1.f);

			float min_f = -1.f;
			float max_f = 0.5f;

			Vec2f v0 = Vec2f(-10, 0.5);

			Vec2f out = clamp(v0, min_v, max_v);	Assert::IsTrue(out.x == 0.f && out.y == 0.5f);
			out = clamp(v0, min_f, max_f);			Assert::IsTrue(out.x == -1.f && out.y == 0.5f);
		}

		TEST_METHOD(TestMax_0)
		{
			Vec4f v0, v1, v2;
			float s;

			v0 = Vec4f(-1, 0, 1, -1);
			v1 = Vec4f(-1, 1, 0, 0);

			v2 = max(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 1.f && v2.z == 1.f && v2.w == 0.f);

			s = 22.f;
			v2 = max(v0, s);
			Assert::IsTrue(v2.x == 22.f && v2.y == 22.f && v2.z == 22.f && v2.w == 22.f);

			s = -2.f;
			v2 = max(s, v0);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f && v2.w == -1.f);
		}
		TEST_METHOD(TestMax_1)
		{
			Vec3f v0, v1, v2;
			float s;

			v0 = Vec3f(-1, 0, 1);
			v1 = Vec3f(-1, 1, 0);

			v2 = max(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 1.f && v2.z == 1.f);

			s = 22.f;
			v2 = max(v0, s);
			Assert::IsTrue(v2.x == 22.f && v2.y == 22.f && v2.z == 22.f);

			s = -2.f;
			v2 = max(s, v0);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f);
		}
		TEST_METHOD(TestMax_2)
		{
			Vec2f v0, v1, v2;
			float s;

			v0 = Vec2f(-1, 0);
			v1 = Vec2f(-1, 1);

			v2 = max(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 1.f);

			s = 22.f;
			v2 = max(v0, s);
			Assert::IsTrue(v2.x == 22.f && v2.y == 22.f);

			s = -2.f;
			v2 = max(s, v0);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f);
		}

		TEST_METHOD(TestMin_0)
		{
			Vec4f v0, v1, v2;
			float s;

			v0 = Vec4f(-1, 0, 1, -1);
			v1 = Vec4f(-1, 1, 0, 0);

			v2 = min(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 0.f && v2.w == -1.f);

			s = 22.f;
			v2 = min(v0, s);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f && v2.w == -1.f);

			s = -2.f;
			v2 = min(s, v0);
			Assert::IsTrue(v2.x == -2.f && v2.y == -2.f && v2.z == -2.f && v2.w == -2.f);
		}
		TEST_METHOD(TestMin_1)
		{
			Vec3f v0, v1, v2;
			float s;

			v0 = Vec3f(-1, 0, 1);
			v1 = Vec3f(-1, 1, 0);

			v2 = min(v0, v1);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 0.f);

			s = 22.f;
			v2 = min(v0, s);
			Assert::IsTrue(v2.x == -1.f && v2.y == 0.f && v2.z == 1.f);

			s = -2.f;
			v2 = min(s, v0);
			Assert::IsTrue(v2.x == -2.f && v2.y == -2.f && v2.z == -2.f);
		}
		TEST_METHOD(TestMin_2)
		{
			Vec2f v0, v1, v2;
			float s;

			v0 = Vec2f(-1, 0);
			v1 = Vec2f(-1, 1);

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
			Vec4f v0; v0 = Vec4f(1, 2, 3, 4); v0 = pow(v0, 2.f); Assert::IsTrue(v0.x == 1.f && v0.y == 4.f && v0.z == 9.f && v0.w == 16.f);
			Vec3f v1; v1 = Vec3f(1, 2, 3); v1 = pow(v1, 2.f); Assert::IsTrue(v1.x == 1.f && v1.y == 4.f && v1.z == 9.f);
			Vec2f v2; v2 = Vec2f(1, 2); v2 = pow(v2, 2.f); Assert::IsTrue(v2.x == 1.f && v2.y == 4.f);
		}

		TEST_METHOD(TestLength_0)
		{
			Vec4f v0;
			float out;

			v0 = Vec4f(0);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));

			v0 = Vec4f(0.182574, 0.365148, 0.547723, 0.730297);
			out = 0.001f * round(length(v0) / 0.001f);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = Vec4f(1, 2, 3, 4);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 5.47723f));
		}
		TEST_METHOD(TestLength_1)
		{
			Vec3f v0;
			float out;

			v0 = Vec3f(0);
			out = nearest(length(v0), 0.001f);
			Assert::IsTrue(out == 0.f);

			v0 = Vec3f(0.267261, 0.534522, 0.801783);
			out = nearest(length(v0), 0.001f);
			Assert::IsTrue(out == 1.f);

			v0 = Vec3f(1, 2, 3);
			out = nearest(length(v0), 0.001f);
			Assert::IsTrue(out == 3.742f);
		}
		TEST_METHOD(TestLength_2)
		{
			Vec2f v0;
			float out;

			v0 = Vec2f(0);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 0.f));

			v0 = Vec2f(0.447213, 0.894427);
			out = 0.001f * round(length(v0) / 0.001f);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 1.f));

			v0 = Vec2f(1, 2);
			out = length(v0);
			Assert::IsTrue(trunc(1000. * out) == trunc(1000. * 2.23606f));
		}

		TEST_METHOD(TestDistance_0)
		{
			Vec4f v0, v1;
			float d;

			v0 = Vec4f(1, 2, 3, 4); v1 = Vec4f(5, 6, 7, 8); d = distance(v0, v1); Assert::IsTrue(d == 8.f);
			v0 = Vec4f(1, 2, 3, 4); v1 = Vec4f(1, 2, 3, 4); d = distance(v0, v1); Assert::IsTrue(d == 0.f);
			v0 = Vec4f(1, 2, 3, 4); v1 = Vec4f(-6, -7, -8, -9); d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 20.4939f);
		}
		TEST_METHOD(TestDistance_1)
		{
			Vec3f v0, v1;
			float d;

			v0 = Vec3f(1, 2, 3); v1 = Vec3f(5, 6, 7);	 d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 6.9282f);
			v0 = Vec3f(1, 2, 3); v1 = Vec3f(1, 2, 3);	 d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 0.f);
			v0 = Vec3f(1, 2, 3); v1 = Vec3f(-6, -7, -8); d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 15.8430f);
		}
		TEST_METHOD(TestDistance_2)
		{
			Vec2f v0, v1;
			float d;

			v0 = Vec2f(2, 3); v1 = Vec2f(5, 6);   d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 4.2426f);
			v0 = Vec2f(2, 3); v1 = Vec2f(2, 3);	  d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 0.f);
			v0 = Vec2f(2, 3); v1 = Vec2f(-6, -7); d = distance(v0, v1); Assert::IsTrue(nearest(d, 0.0001f) == 12.8062f);
		}

		TEST_METHOD(TestNormalize)
		{
			//Assert::IsTrue(normalize(Vec4f(1,2,3,4)) == Vec4f(0.5));
			Assert::IsTrue(length(normalize(Vec4f(1, 2, 3, 4))) == 1.f);
			Assert::IsTrue(normalize(Vec4f(2)) == Vec4f(0.5));
			Assert::IsTrue(length(normalize(Vec4f(2))) == 1.f);
			Assert::IsTrue(normalize(Vec4f(0.f)) == Vec4f(0.f));

			//Assert::IsTrue(normalize(Vec3f(1,2,3)) == Vec3f());
			Assert::IsTrue(length(normalize(Vec3f(1, 2, 3))) == 1.f);
			//Assert::IsTrue(normalize(Vec3f(2)) == Vec3f(0.5));
			Assert::IsTrue(length(normalize(Vec3f(2))) == 1.f);
			Assert::IsTrue(normalize(Vec3f(0.f)) == Vec3f(0.f));

			//Assert::IsTrue(normalize(Vec2f(1,2)) == Vec3f());
			Assert::IsTrue(length(normalize(Vec2f(1, 3))) == 1.f);
			Assert::IsTrue(normalize(Vec2f(10, 0)) == Vec2f(1, 0));
			Assert::IsTrue(length(normalize(Vec2f(10, 0))) == 1.f);
			Assert::IsTrue(normalize(Vec2f(0.f)) == Vec2f(0.f));
		}

		TEST_METHOD(TestReflect)
		{
			Assert::IsTrue(false);
		}

		TEST_METHOD(TestCross)
		{
			Assert::IsTrue(cross(Vec3f(1, 2, 3), Vec3f(3, 4, 5)) == Vec3f(-2, 4, -2));
			Assert::IsTrue(cross(Vec3f(-1, 5, -20), Vec3f(3, 4, 5)) == Vec3f(105, -55, -19));
			Assert::IsTrue(cross(Vec3f(0, 1, 0), Vec3f(-1, 0, 0)) == Vec3f(0, 0, 1));
			Assert::IsTrue(cross(Vec3f(0), Vec3f(0)) == Vec3f(0));
			Assert::IsTrue(cross(Vec3f(1), Vec3f(1)) == Vec3f(0));
		}

		TEST_METHOD(TestMix)
		{
			Assert::IsTrue(false);
		}
	};
}
