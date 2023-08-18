#include <gtest/gtest.h>
#include <vector>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtx/hash.hpp>

#include "nonlinear_solver.hpp"

namespace
{
    struct square_func
    {
        template <typename T>
        T operator()(T *x) const
        {
            return x[0] * x[0];
        }
    };
    struct cubic_func
    {
        template <typename T>
        T operator()(T *x) const
        {
            return x[0] * x[0] * x[0];
        }
    };
}

TEST(SolverTest, TestAutoDiff)
{
    const auto param1 = std::make_shared<optimization::parameter_block>(0.0);
    param1->offset = 0;
    const auto param2 = std::make_shared<optimization::parameter_block>(0.0);
    param2->offset = 1;

    const auto term1 = optimization::make_auto_diff_function_term(square_func(), param1);
    const auto term2 = optimization::make_auto_diff_function_term(cubic_func(), param2);

    optimization::function func(term1, term2);

    {
        std::vector<double> params = {2, 3};
        const auto value = func.eval(params.data(), params.size());
        ASSERT_NEAR(value, 31, 1e-9);
    }

    {
        std::vector<double> params = {2, 3};
        std::vector<double> grad(2);
        func.grad(params.data(), params.size(), grad.data());
        ASSERT_NEAR(grad[0], 4, 1e-9);
        ASSERT_NEAR(grad[1], 27, 1e-9);
    }

    {
        std::vector<double> params = {2, 3};
        std::vector<double> grad(2);
        std::vector<double> hessian(4);
        func.hessian(params.data(), params.size(), grad.data(), hessian.data());
        ASSERT_NEAR(hessian[0], 2, 1e-9);
        ASSERT_NEAR(hessian[1], 0, 1e-9);
        ASSERT_NEAR(hessian[2], 0, 1e-9);
        ASSERT_NEAR(hessian[3], 18, 1e-9);
    }
}

TEST(SolverTest, TestVec3f)
{
    vec3_t<float> vec1(1, 2, 3);

    ASSERT_FLOAT_EQ(vec1.x, 1.0f);
    ASSERT_FLOAT_EQ(vec1.y, 2.0f);
    ASSERT_FLOAT_EQ(vec1.z, 3.0f);

    vec3_t<float> vec2(4, 5, 6);

    {
        const auto vec3 = vec1 + vec2;

        ASSERT_FLOAT_EQ(vec3.x, 5.0f);
        ASSERT_FLOAT_EQ(vec3.y, 7.0f);
        ASSERT_FLOAT_EQ(vec3.z, 9.0f);
    }
}

TEST(SolverTest, TestVec3d)
{
    vec3_t<double> vec1(1, 2, 3);

    ASSERT_DOUBLE_EQ(vec1.x, 1.0);
    ASSERT_DOUBLE_EQ(vec1.y, 2.0);
    ASSERT_DOUBLE_EQ(vec1.z, 3.0);

    vec3_t<double> vec2(4, 5, 6);

    {
        const auto vec3 = -vec1;

        ASSERT_DOUBLE_EQ(vec3.x, -1.0);
        ASSERT_DOUBLE_EQ(vec3.y, -2.0);
        ASSERT_DOUBLE_EQ(vec3.z, -3.0);
    }

    {
        const auto vec3 = vec1 + vec2;

        ASSERT_DOUBLE_EQ(vec3.x, 5.0);
        ASSERT_DOUBLE_EQ(vec3.y, 7.0);
        ASSERT_DOUBLE_EQ(vec3.z, 9.0);
    }

    {
        const auto vec3 = vec1 - vec2;

        ASSERT_DOUBLE_EQ(vec3.x, -3.0);
        ASSERT_DOUBLE_EQ(vec3.y, -3.0);
        ASSERT_DOUBLE_EQ(vec3.z, -3.0);
    }

    {
        const auto value = dot(vec1, vec2);
        ASSERT_DOUBLE_EQ(value, 32.0);
    }
}

TEST(SolverTest, TestDualVec3d)
{
    vec3_t<dual_t<double>> vec1(dual_t<double>(1, 2), dual_t<double>(3, 4), dual_t<double>(5, 6));

    vec3_t<dual_t<double>> vec2(dual_t<double>(7, 8), dual_t<double>(9, 10), dual_t<double>(11, 12));

    {
        const auto value = dot(vec1, vec2);

        dual_t<double> vec1_x(vec1.a.x, vec1.b.x);
        dual_t<double> vec1_y(vec1.a.y, vec1.b.y);
        dual_t<double> vec1_z(vec1.a.z, vec1.b.z);
        dual_t<double> vec2_x(vec2.a.x, vec2.b.x);
        dual_t<double> vec2_y(vec2.a.y, vec2.b.y);
        dual_t<double> vec2_z(vec2.a.z, vec2.b.z);
        const auto expected = vec1_x * vec2_x + vec1_y * vec2_y + vec1_z * vec2_z;
        ASSERT_DOUBLE_EQ(value.a, expected.a);
        ASSERT_DOUBLE_EQ(value.b, expected.b);
    }

    {
        const auto value = cross(vec1, vec2);

        dual_t<double> vec1_x(vec1.a.x, vec1.b.x);
        dual_t<double> vec1_y(vec1.a.y, vec1.b.y);
        dual_t<double> vec1_z(vec1.a.z, vec1.b.z);
        dual_t<double> vec2_x(vec2.a.x, vec2.b.x);
        dual_t<double> vec2_y(vec2.a.y, vec2.b.y);
        dual_t<double> vec2_z(vec2.a.z, vec2.b.z);

        const auto expected_x = vec1_y * vec2_z - vec1_z * vec2_y;
        const auto expected_y = vec1_z * vec2_x - vec1_x * vec2_z;
        const auto expected_z = vec1_x * vec2_y - vec1_y * vec2_x;
        ASSERT_DOUBLE_EQ(value.a.x, expected_x.a);
        ASSERT_DOUBLE_EQ(value.a.y, expected_y.a);
        ASSERT_DOUBLE_EQ(value.a.z, expected_z.a);
        ASSERT_DOUBLE_EQ(value.b.x, expected_x.b);
        ASSERT_DOUBLE_EQ(value.b.y, expected_y.b);
        ASSERT_DOUBLE_EQ(value.b.z, expected_z.b);
    }
}
