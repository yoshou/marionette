#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include "automatic_differentiation.hpp"

// Helper functions for differentiation tests
template <typename T>
auto test_func1(const T &x)
{
    return x * x * x * 4.0 + x * x * 5.0 + x * 6.0 + 10.0;
}

template <typename T>
auto test_dfunc1(const T &x)
{
    return x * x * 3.0 * 4.0 + x * 10.0 + 6.0;
}

// Test automatic differentiation with dual_t and hyper_dual_t
TEST(AutoDiffTest, BasicDifferentiation)
{
    // Test dual_t with default initialization (derivative of constant)
    {
        const auto x = dual_t<double>(2.0);
        const auto y = test_func1(x);
        // f(2) = 4*2³ + 5*2² + 6*2 + 10 = 32 + 20 + 12 + 10 = 74
        ASSERT_DOUBLE_EQ(y.a, 74.0);
        ASSERT_DOUBLE_EQ(y.b, 0.0);   // df/dx = 0 when dual part is 0
    }
    
    // Test dual_t with unit derivative
    {
        const auto x = dual_t<double>(2.0, 1.0);
        const auto y = test_func1(x);
        const auto expected_deriv = test_dfunc1(2.0);
        // f'(2) = 12*2² + 10*2 + 6 = 48 + 20 + 6 = 74
        ASSERT_DOUBLE_EQ(y.a, 74.0);
        ASSERT_DOUBLE_EQ(y.b, expected_deriv);
    }
    
    // Test hyper_dual_t
    {
        const auto x = hyper_dual_t<double>(2.0);
        const auto y = test_func1(x);
        ASSERT_DOUBLE_EQ(y.a, 74.0);
        ASSERT_DOUBLE_EQ(y.b, 0.0);
        ASSERT_DOUBLE_EQ(y.c, 0.0);
        ASSERT_DOUBLE_EQ(y.d, 0.0);
    }
    
    // Test hyper_dual_t with derivatives
    {
        const auto x = hyper_dual_t<double>(2.0, 1.0, 0.0, 0.0);
        const auto y = test_func1(x);
        ASSERT_DOUBLE_EQ(y.a, 74.0);
        ASSERT_NEAR(y.b, 74.0, 1e-9);  // First derivative
    }
    
    // Test dual_t applied to derivative function
    {
        const auto x = dual_t<double>(2.0, 1.0);
        const auto y = test_dfunc1(x);
        // f'(x) = 12x² + 10x + 6
        // f'(2) = 74
        // f''(x) = 24x + 10
        // f''(2) = 58
        ASSERT_DOUBLE_EQ(y.a, 74.0);
        ASSERT_DOUBLE_EQ(y.b, 58.0);
    }
}

// Test dual_t comparison and special functions
TEST(AutoDiffTest, DualComparisons)
{
    dual_t<double> a(3.0, 1.0);
    dual_t<double> b(5.0, 2.0);
    
    ASSERT_TRUE(b > a);
    ASSERT_FALSE(a > b);
}

TEST(AutoDiffTest, DualSpecialFunctions)
{
    // Test isinf
    {
        dual_t<double> finite(3.0, 1.0);
        dual_t<double> inf_val(std::numeric_limits<double>::infinity(), 1.0);
        
        ASSERT_FALSE(isinf(finite));
        ASSERT_TRUE(isinf(inf_val));
    }
    
    // Test isnan
    {
        dual_t<double> normal(3.0, 1.0);
        dual_t<double> nan_val(std::numeric_limits<double>::quiet_NaN(), 1.0);
        
        ASSERT_FALSE(isnan(normal));
        ASSERT_TRUE(isnan(nan_val));
    }
    
    // Test isless
    {
        dual_t<double> a(3.0, 1.0);
        dual_t<double> b(5.0, 2.0);
        
        ASSERT_TRUE(isless(a, b));
        ASSERT_FALSE(isless(b, a));
    }
    
    // Test isgreater
    {
        dual_t<double> a(3.0, 1.0);
        dual_t<double> b(5.0, 2.0);
        
        ASSERT_TRUE(isgreater(b, a));
        ASSERT_FALSE(isgreater(a, b));
    }
    
    // Test islessgreater
    {
        dual_t<double> a(3.0, 1.0);
        dual_t<double> b(5.0, 2.0);
        dual_t<double> c(3.0, 2.0);
        
        ASSERT_TRUE(islessgreater(a, b));
        ASSERT_FALSE(islessgreater(a, c));
    }
}

TEST(AutoDiffTest, DualFmin)
{
    // Normal case
    {
        dual_t<double> a(3.0, 1.0);
        dual_t<double> b(5.0, 2.0);
        
        auto result = fmin(a, b);
        ASSERT_DOUBLE_EQ(result.a, 3.0);
        ASSERT_DOUBLE_EQ(result.b, 1.0);
    }
    
    // Equal values
    {
        dual_t<double> a(4.0, 1.0);
        dual_t<double> b(4.0, 2.0);
        
        auto result = fmin(a, b);
        // Should return average when equal
        ASSERT_DOUBLE_EQ(result.a, 4.0);
        ASSERT_DOUBLE_EQ(result.b, 1.5);
    }
    
    // NaN handling
    {
        dual_t<double> a(3.0, 1.0);
        dual_t<double> nan_val(std::numeric_limits<double>::quiet_NaN(), 1.0);
        
        auto result = fmin(a, nan_val);
        ASSERT_DOUBLE_EQ(result.a, 3.0);
        ASSERT_DOUBLE_EQ(result.b, 1.0);
    }
}

// Test hyper_dual_t special functions
TEST(AutoDiffTest, HyperDualSpecialFunctions)
{
    // Test isinf
    {
        hyper_dual_t<double> finite(3.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> inf_val(std::numeric_limits<double>::infinity(), 1.0, 2.0, 0.5);
        
        ASSERT_FALSE(isinf(finite));
        ASSERT_TRUE(isinf(inf_val));
    }
    
    // Test isnan
    {
        hyper_dual_t<double> normal(3.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> nan_val(std::numeric_limits<double>::quiet_NaN(), 1.0, 2.0, 0.5);
        
        ASSERT_FALSE(isnan(normal));
        ASSERT_TRUE(isnan(nan_val));
    }
    
    // Test isless
    {
        hyper_dual_t<double> a(3.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> b(5.0, 2.0, 3.0, 1.0);
        
        ASSERT_TRUE(isless(a, b));
        ASSERT_FALSE(isless(b, a));
    }
    
    // Test isgreater
    {
        hyper_dual_t<double> a(3.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> b(5.0, 2.0, 3.0, 1.0);
        
        ASSERT_TRUE(isgreater(b, a));
        ASSERT_FALSE(isgreater(a, b));
    }
    
    // Test islessgreater
    {
        hyper_dual_t<double> a(3.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> b(5.0, 2.0, 3.0, 1.0);
        hyper_dual_t<double> c(3.0, 2.0, 1.0, 0.3);
        
        ASSERT_TRUE(islessgreater(a, b));
        ASSERT_FALSE(islessgreater(a, c));
    }
}

TEST(AutoDiffTest, HyperDualFmin)
{
    // Normal case
    {
        hyper_dual_t<double> a(3.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> b(5.0, 2.0, 3.0, 1.0);
        
        auto result = fmin(a, b);
        ASSERT_DOUBLE_EQ(result.a, 3.0);
        ASSERT_DOUBLE_EQ(result.b, 1.0);
    }
    
    // Equal values
    {
        hyper_dual_t<double> a(4.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> b(4.0, 2.0, 3.0, 1.0);
        
        auto result = fmin(a, b);
        // Should return average when equal
        ASSERT_DOUBLE_EQ(result.a, 4.0);
        ASSERT_DOUBLE_EQ(result.b, 1.5);
    }
    
    // NaN handling
    {
        hyper_dual_t<double> a(3.0, 1.0, 2.0, 0.5);
        hyper_dual_t<double> nan_val(std::numeric_limits<double>::quiet_NaN(), 1.0, 2.0, 0.5);
        
        auto result = fmin(a, nan_val);
        ASSERT_DOUBLE_EQ(result.a, 3.0);
        ASSERT_DOUBLE_EQ(result.b, 1.0);
    }
}

// Test dual_vec_t operations
TEST(AutoDiffTest, DualVecOperations)
{
    dual_vec_t<double, 3> a(2.0, 0);
    dual_vec_t<double, 3> b(3.0, 1);
    
    // Test addition
    {
        auto c = a + b;
        ASSERT_DOUBLE_EQ(c.a, 5.0);
        ASSERT_DOUBLE_EQ(c.v[0], 1.0);
        ASSERT_DOUBLE_EQ(c.v[1], 1.0);
        ASSERT_DOUBLE_EQ(c.v[2], 0.0);
    }
    
    // Test subtraction
    {
        auto c = a - b;
        ASSERT_DOUBLE_EQ(c.a, -1.0);
        ASSERT_DOUBLE_EQ(c.v[0], 1.0);
        ASSERT_DOUBLE_EQ(c.v[1], -1.0);
        ASSERT_DOUBLE_EQ(c.v[2], 0.0);
    }
    
    // Test multiplication
    {
        auto c = a * b;
        ASSERT_DOUBLE_EQ(c.a, 6.0);
        ASSERT_DOUBLE_EQ(c.v[0], 3.0);
        ASSERT_DOUBLE_EQ(c.v[1], 2.0);
        ASSERT_DOUBLE_EQ(c.v[2], 0.0);
    }
    
    // Test division
    {
        auto c = a / b;
        ASSERT_NEAR(c.a, 2.0/3.0, 1e-9);
    }
    
    // Test sqrt
    {
        dual_vec_t<double, 2> d(4.0, 0);
        auto e = sqrt(d);
        ASSERT_DOUBLE_EQ(e.a, 2.0);
        ASSERT_DOUBLE_EQ(e.v[0], 0.25);
    }
    
    // Test sin/cos
    {
        dual_vec_t<double, 1> f(0.0, 0);
        auto s = sin(f);
        auto c = cos(f);
        ASSERT_NEAR(s.a, 0.0, 1e-9);
        ASSERT_NEAR(c.a, 1.0, 1e-9);
    }
}

// Test vec3_t operations
TEST(AutoDiffTest, Vec3Operations)
{
    // Test float specialization
    {
        vec3_t<float> v1(1.0f, 2.0f, 3.0f);
        vec3_t<float> v2(4.0f, 5.0f, 6.0f);
        
        auto v3 = v1 + v2;
        ASSERT_FLOAT_EQ(v3.x, 5.0f);
        ASSERT_FLOAT_EQ(v3.y, 7.0f);
        ASSERT_FLOAT_EQ(v3.z, 9.0f);
        
        auto d = dot(v1, v2);
        ASSERT_FLOAT_EQ(d, 32.0f);
    }
    
    // Test double specialization
    {
        vec3_t<double> v1(1.0, 2.0, 3.0);
        vec3_t<double> v2(4.0, 5.0, 6.0);
        
        auto v3 = v1 * v2;
        ASSERT_DOUBLE_EQ(v3.x, 4.0);
        ASSERT_DOUBLE_EQ(v3.y, 10.0);
        ASSERT_DOUBLE_EQ(v3.z, 18.0);
        
        auto v4 = cross(v1, v2);
        ASSERT_DOUBLE_EQ(v4.x, -3.0);
        ASSERT_DOUBLE_EQ(v4.y, 6.0);
        ASSERT_DOUBLE_EQ(v4.z, -3.0);
    }
}

// Test quaternion rotation
TEST(AutoDiffTest, QuaternionRotation)
{
    // Identity quaternion
    quat_t<double> q_identity(1.0, 0.0, 0.0, 0.0);
    vec3_t<double> v(1.0, 0.0, 0.0);
    
    auto result = rotate(q_identity, v);
    ASSERT_NEAR(result.x, 1.0, 1e-9);
    ASSERT_NEAR(result.y, 0.0, 1e-9);
    ASSERT_NEAR(result.z, 0.0, 1e-9);
    
    // 90 degree rotation around z-axis
    double angle = M_PI / 2.0;
    quat_t<double> q_90z(std::cos(angle/2), 0.0, 0.0, std::sin(angle/2));
    auto result2 = rotate(q_90z, v);
    ASSERT_NEAR(result2.x, 0.0, 1e-9);
    ASSERT_NEAR(result2.y, 1.0, 1e-9);
    ASSERT_NEAR(result2.z, 0.0, 1e-9);
}

TEST(AutoDiffTest, QuaternionRotationDual)
{
    // Test with dual numbers
    quat_t<dual_t<double>> q(
        dual_t<double>(1.0, 0.0),
        dual_t<double>(0.0, 0.0),
        dual_t<double>(0.0, 0.0),
        dual_t<double>(0.0, 0.0)
    );
    
    vec3_t<dual_t<double>> v(
        dual_t<double>(1.0, 0.0),
        dual_t<double>(0.0, 0.0),
        dual_t<double>(0.0, 0.0)
    );
    
    auto result = rotate(q, v);
    ASSERT_NEAR(result.a.x, 1.0, 1e-9);
    ASSERT_NEAR(result.a.y, 0.0, 1e-9);
    ASSERT_NEAR(result.a.z, 0.0, 1e-9);
}

TEST(AutoDiffTest, RotateAngleAxis)
{
    // Zero rotation
    {
        vec3_t<double> axis(0.0, 0.0, 0.0);
        vec3_t<double> point(1.0, 0.0, 0.0);
        
        auto result = rotate_angle_axis(axis, point);
        ASSERT_NEAR(result.x, 1.0, 1e-9);
        ASSERT_NEAR(result.y, 0.0, 1e-9);
        ASSERT_NEAR(result.z, 0.0, 1e-9);
    }
    
    // 90 degree rotation around z-axis
    {
        vec3_t<double> axis(0.0, 0.0, M_PI / 2.0);
        vec3_t<double> point(1.0, 0.0, 0.0);
        
        auto result = rotate_angle_axis(axis, point);
        ASSERT_NEAR(result.x, 0.0, 1e-9);
        ASSERT_NEAR(result.y, 1.0, 1e-9);
        ASSERT_NEAR(result.z, 0.0, 1e-9);
    }
}

// Test compound operations
TEST(AutoDiffTest, CompoundOperations)
{
    dual_t<double> x(2.0, 1.0);
    
    // Test pow
    auto y = pow(x, 3.0);
    ASSERT_DOUBLE_EQ(y.a, 8.0);
    ASSERT_DOUBLE_EQ(y.b, 12.0);
    
    // Test square
    auto z = square(x);
    ASSERT_DOUBLE_EQ(z.a, 4.0);
    ASSERT_DOUBLE_EQ(z.b, 4.0);
}

// Test assignment operators
TEST(AutoDiffTest, AssignmentOperators)
{
    // dual_t
    {
        dual_t<double> a(2.0, 1.0);
        dual_t<double> b(3.0, 2.0);
        
        a += b;
        ASSERT_DOUBLE_EQ(a.a, 5.0);
        ASSERT_DOUBLE_EQ(a.b, 3.0);
        
        a -= b;
        ASSERT_DOUBLE_EQ(a.a, 2.0);
        ASSERT_DOUBLE_EQ(a.b, 1.0);
        
        a *= b;
        ASSERT_DOUBLE_EQ(a.a, 6.0);
        ASSERT_DOUBLE_EQ(a.b, 7.0);
        
        a /= b;
        ASSERT_DOUBLE_EQ(a.a, 2.0);
        ASSERT_DOUBLE_EQ(a.b, 1.0);
        
        a += 5.0;
        ASSERT_DOUBLE_EQ(a.a, 7.0);
        ASSERT_DOUBLE_EQ(a.b, 1.0);
    }
    
    // hyper_dual_t
    {
        hyper_dual_t<double> a(2.0, 1.0, 1.0, 0.0);
        hyper_dual_t<double> b(3.0, 2.0, 2.0, 1.0);
        
        a += b;
        ASSERT_DOUBLE_EQ(a.a, 5.0);
        ASSERT_DOUBLE_EQ(a.b, 3.0);
        
        a -= b;
        ASSERT_DOUBLE_EQ(a.a, 2.0);
        ASSERT_DOUBLE_EQ(a.b, 1.0);
    }
}
