#pragma once

#include <vector>
#include <algorithm>
#include <glm/glm.hpp>

#include <Eigen/QR>

static void polyfit(const std::vector<double> &t,
             const std::vector<double> &v,
             std::vector<double> &coeff,
             int order

)
{
    // Create Matrix Placeholder of size n x k, n= number of datapoints, k = order of polynomial, for exame k = 3 for cubic polynomial
    Eigen::MatrixXd T(t.size(), order + 1);
    Eigen::VectorXd V = Eigen::VectorXd::Map(&v.front(), v.size());
    Eigen::VectorXd result;

    // check to make sure inputs are correct
    assert(t.size() == v.size());
    assert(t.size() >= order + 1);
    // Populate the matrix
    for (size_t i = 0; i < t.size(); ++i)
    {
        for (size_t j = 0; j < order + 1; ++j)
        {
            T(i, j) = pow(t.at(i), j);
        }
    }

    // Solve for linear least square fit
    result = T.householderQr().solve(V);
    coeff.resize(order + 1);
    for (int k = 0; k < order + 1; k++)
    {
        coeff[k] = result[k];
    }
}

struct polynomial_regression
{
    std::vector<glm::vec3> values;
    std::size_t order;

    polynomial_regression(std::size_t n, std::size_t order = 2)
        : values(n), order(order)
    {}

    void update(glm::vec3 value)
    {
        for (std::size_t i = values.size() - 1; i > 0; i--)
        {
            values[i] = values[i - 1];
        }
        values[0] = value;
    }

    glm::vec3 predict() const
    {
        std::vector<double> x_values;
        std::vector<double> y_values;
        std::vector<double> z_values;
        std::vector<double> t_values;
        std::transform(values.begin(), values.end(), std::back_inserter(x_values), [](glm::vec3 value)
                       { return value.x; });
        std::transform(values.begin(), values.end(), std::back_inserter(y_values), [](glm::vec3 value)
                       { return value.y; });
        std::transform(values.begin(), values.end(), std::back_inserter(z_values), [](glm::vec3 value)
                       { return value.z; });
        for (std::size_t i = 0; i < values.size(); i++)
        {
            t_values.push_back(-static_cast<float>(i));
        }

        std::vector<double> x_coeffs;
        std::vector<double> y_coeffs;
        std::vector<double> z_coeffs;
        polyfit(t_values, x_values, x_coeffs, order);
        polyfit(t_values, y_values, y_coeffs, order);
        polyfit(t_values, z_values, z_coeffs, order);

        const auto t = static_cast<float>(1);

        double x_value = 0;
        double y_value = 0;
        double z_value = 0;
        double s = 1;
        for (std::size_t i = 0; i <= order; i++)
        {
            x_value += x_coeffs[i] * s;
            y_value += y_coeffs[i] * s;
            z_value += z_coeffs[i] * s;
            s *= t;
        }

        return glm::vec3(x_value, y_value, z_value);
    }
};
