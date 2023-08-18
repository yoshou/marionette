#pragma once

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

static void test_automatic_differentiation()
{
    {
        const auto x = dual_t<double>(2.0);
        const auto y = test_func1(x);
        std::cout << y.a << ", " << y.b << std::endl;
    }
    {
        const auto x = 2.0;
        const auto y = test_dfunc1(x);
        std::cout << y << std::endl;
    }

    {
        const auto x = hyper_dual_t<double>(2.0);
        const auto y = test_func1(x);
        std::cout << y.a << ", " << y.b << ", " << y.c << ", " << y.d << std::endl;
    }
    {
        const auto x = dual_t<double>(2.0);
        const auto y = test_dfunc1(x);
        std::cout << y.a << ", " << y.b << std::endl;
    }
}