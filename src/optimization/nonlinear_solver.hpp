#pragma once

#include <chrono>
#include <numeric>
#include <memory>
#include <cstdint>
#include <iostream>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include "automatic_differentiation.hpp"

#include "unsupported/Eigen/NonLinearOptimization"

template <typename ResidualFunc, typename ConstraintFunc, typename T>
static void gradient(ResidualFunc **residuals, std::size_t num_residuals, ConstraintFunc **constraints, std::size_t num_constraints, const T *params, std::size_t num_params, T *nabla)
{
    std::vector<dual_t<T>> v(num_params);

    for (std::size_t i = 0; i < num_params; i++)
    {
        for (std::size_t k = 0; k < num_params; k++)
        {
            v[k] = dual_t<T>(params[k], T{0});
        }
        v[i].b = T{1};

        dual_t<T> result(T{0});
        for (std::size_t k = 0; k < num_residuals; k++)
        {
            result += (*(residuals[k]))(v.data());
        }
        for (std::size_t k = 0; k < num_constraints; k++)
        {
            result += (*constraints[k])(v.data());
        }

        nabla[i] = result.b;
    }
}

template <typename ResidualFunc, typename ConstraintFunc, typename T>
static void hessian(ResidualFunc **residuals, std::size_t num_residuals, ConstraintFunc **constraints, std::size_t num_constraints, const T *params, std::size_t num_params, T *nabla, T *H)
{
    std::vector<hyper_dual_t<T>> v(num_params);

    for (std::size_t i = 0; i < num_params; i++)
    {
        for (std::size_t k = 0; k < num_params; k++)
        {
            v[k] = hyper_dual_t<T>(params[k], T{0}, T{0}, T{0});
        }
        v[i].b = T{1};
        v[i].c = T{1};

        hyper_dual_t<T> result(T{0});
        for (std::size_t k = 0; k < num_residuals; k++)
        {
            result += (*(residuals[k]))(v.data());
        }
        for (std::size_t k = 0; k < num_constraints; k++)
        {
            result += (*constraints[k])(v.data());
        }

        nabla[i] = result.b;
        H[i * num_params + i] = result.d;

        for (std::size_t j = i + 1; j < num_params; j++)
        {
            for (std::size_t k = 0; k < num_params; k++)
            {
                v[k] = hyper_dual_t<T>(params[k], T{0}, T{0}, T{0});
            }
            v[i].b = T{1};
            v[j].c = T{1};

            hyper_dual_t<T> result(T{0});
            for (std::size_t k = 0; k < num_residuals; k++)
            {
                result += (*residuals[k])(v.data());
            }
            for (std::size_t k = 0; k < num_constraints; k++)
            {
                result += (*constraints[k])(v.data());
            }

            H[i * num_params + j] = result.d;
            H[j * num_params + i] = result.d;
        }
    }
}

template <typename Func, typename T>
static void hessian(Func **funcs, std::size_t num_funcs, const T *params, std::size_t num_params, T *nabla, T *H)
{
    std::vector<hyper_dual_t<T>> v(num_params);

    for (std::size_t i = 0; i < num_params; i++)
    {
        for (std::size_t k = 0; k < num_params; k++)
        {
            v[k] = hyper_dual_t<T>(params[k], T{0}, T{0}, T{0});
        }
        v[i].b = T{1};
        v[i].c = T{1};

        hyper_dual_t<T> result(T{0});
        for (std::size_t k = 0; k < num_funcs; k++)
        {
            result += (*(funcs[k]))(v.data());
        }

        nabla[i] = result.b;
        H[i * num_params + i] = result.d;

        for (std::size_t j = i + 1; j < num_params; j++)
        {
            for (std::size_t k = 0; k < num_params; k++)
            {
                v[k] = hyper_dual_t<T>(params[k], T{0}, T{0}, T{0});
            }
            v[i].b = T{1};
            v[j].c = T{1};

            hyper_dual_t<T> result(T{0});
            for (std::size_t k = 0; k < num_funcs; k++)
            {
                result += (*funcs[k])(v.data());
            }

            H[i * num_params + j] = result.d;
            H[j * num_params + i] = result.d;
        }
    }
}

template <typename Func, typename T>
static void jacobi(Func **funcs, std::size_t num_funcs, const T *params, std::size_t num_params, T *J)
{
    std::vector<dual_t<T>> v(num_params);

    for (std::size_t i = 0; i < num_funcs; i++)
    {
        for (std::size_t j = 0; j < num_params; j++)
        {
            for (std::size_t k = 0; k < num_params; k++)
            {
                v[k] = dual_t<T>(params[k], T{0});
            }
            v[j].b = T{1};

            dual_t<T> result = (*(funcs[i]))(v.data());

            J[i * num_params + j] = result.b;
        }
    }
}

struct optimization_result
{
    double initial_residual_error;
    double initial_error;
    double final_residual_error;
    double final_error;
    double total_elapsed_time;
};

template <typename ResidualFunc, typename ConstraintFunc, typename T>
static optimization_result solve_newton_method(ResidualFunc **residuals, std::size_t num_residuals, ConstraintFunc **constraints, std::size_t num_constraints, T *params, std::size_t num_params, double terminate_thresold, std::size_t max_iteration)
{
    optimization_result result;

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_residuals; i++)
        {
            error += (*(residuals[i]))(params);
        }
        result.initial_residual_error = error;

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            error += (*constraints[i])(params);
        }
        result.initial_error = error;
    }

    const auto start = std::chrono::system_clock::now();

    std::vector<double> H(num_params * num_params);
    std::vector<double> nabla(num_params);
    auto last_error = std::numeric_limits<double>::max();
    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        hessian(residuals, num_residuals, constraints, num_constraints, params, num_params, nabla.data(), H.data());

        Eigen::MatrixXd H_m(num_params, num_params);
        for (std::size_t i = 0; i < num_params; i++)
        {
            for (std::size_t j = 0; j < num_params; j++)
            {
                H_m(i, j) = H[i * num_params + j];
            }
        }

        Eigen::VectorXd nabla_v(num_params);
        for (std::size_t i = 0; i < num_params; i++)
        {
            nabla_v(i) = nabla[i];
        }

        Eigen::VectorXd params_v(num_params);
        for (std::size_t i = 0; i < num_params; i++)
        {
            params_v(i) = params[i];
        }

        const auto next_params_v = params_v - H_m.inverse() * nabla_v;

        for (std::size_t i = 0; i < num_params; i++)
        {
            params[i] = next_params_v(i);
        }

        auto error = 0.0;
        for (std::size_t i = 0; i < num_residuals; i++)
        {
            error += (*residuals[i])(params);
        }
        for (std::size_t i = 0; i < num_constraints; i++)
        {
            error += (*constraints[i])(params);
        }

        std::cout << "Error : " << error << std::endl;

        if ((std::abs(last_error - error) / error) < terminate_thresold)
        {
            break;
        }

        last_error = error;
    }

    const auto end = std::chrono::system_clock::now();
    result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_residuals; i++)
        {
            error += (*(residuals[i]))(params);
        }
        result.final_residual_error = error;

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            error += (*constraints[i])(params);
        }
        result.final_error = error;
    }

    return result;
}

template <typename ResidualFunc, typename ConstraintFunc, typename T>
static T eval(ResidualFunc **residuals, std::size_t num_residuals, ConstraintFunc **constraints, std::size_t num_constraints, const T *params)
{
    auto value = T{0};
    for (std::size_t i = 0; i < num_residuals; i++)
    {
        value += (*(residuals[i]))(params);
    }
    for (std::size_t i = 0; i < num_constraints; i++)
    {
        value += (*constraints[i])(params);
    }
    return value;
}

template <typename Func, typename T>
static T eval(Func **funcs, std::size_t num_funcs, const T *params)
{
    auto value = T{0};
    for (std::size_t i = 0; i < num_funcs; i++)
    {
        value += (*(funcs[i]))(params);
    }
    return value;
}

template <typename ResidualFunc, typename ConstraintFunc>
static double armijo(ResidualFunc **residuals, std::size_t num_residuals, ConstraintFunc **constraints, std::size_t num_constraints,
                     const Eigen::VectorXd &params, const Eigen::VectorXd &grad, const Eigen::VectorXd &d, std::size_t max_iteration = 10000)
{
    auto alpha = 1.0;
    auto rho = 0.5;
    double c1 = 1e-3;

    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        Eigen::VectorXd step_params = params + d *alpha;
        if (eval(residuals, num_residuals, constraints, num_constraints, step_params.data()) <=
            eval(residuals, num_residuals, constraints, num_constraints, params.data()) + c1 * grad.dot(d))
        {
            break;
        }
        else
        {
            alpha = alpha * rho;
        }
    }

    return alpha;
}

template <typename ResidualFunc, typename ConstraintFunc, typename T>
static optimization_result solve_semi_newton_method(ResidualFunc **residuals, std::size_t num_residuals, ConstraintFunc **constraints, std::size_t num_constraints, T *params, std::size_t num_params, double terminate_thresold, std::size_t max_iteration)
{
    optimization_result result;

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_residuals; i++)
        {
            error += (*(residuals[i]))(params);
        }
        result.initial_residual_error = error;

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            error += (*constraints[i])(params);
        }
        result.initial_error = error;
    }

    const auto start = std::chrono::system_clock::now();

    Eigen::MatrixXd B = Eigen::MatrixXd::Identity(num_params, num_params);
    auto last_error = std::numeric_limits<double>::max();

    Eigen::VectorXd params_v(num_params);
    for (std::size_t i = 0; i < num_params; i++)
    {
        params_v(i) = params[i];
    }
    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        Eigen::VectorXd grad_v(num_params);
        gradient(residuals, num_residuals, constraints, num_constraints, params_v.data(), num_params, grad_v.data());

        Eigen::VectorXd d = B.inverse() * -grad_v;

        const auto alpha = armijo(residuals, num_residuals, constraints, num_constraints, params_v, grad_v, d);

        Eigen::VectorXd next_params_v = params_v + alpha * d;

        // Update B
        Eigen::VectorXd next_grad_v(num_params);
        gradient(residuals, num_residuals, constraints, num_constraints, next_params_v.data(), num_params, next_grad_v.data());

        Eigen::VectorXd s_v = alpha * d;
        Eigen::VectorXd dgrad_v = next_grad_v - grad_v;

        Eigen::VectorXd Bs_v = B * s_v;
        const auto sdgrad = s_v.dot(dgrad_v);
        const auto sBs = s_v.dot(Bs_v);

        Eigen::MatrixXd BsBs_m = Bs_v * Bs_v.transpose();
        Eigen::MatrixXd dgrad_dgrad_m = dgrad_v * dgrad_v.transpose();

        B = B - BsBs_m / sBs + dgrad_dgrad_m / sdgrad;
        params_v = next_params_v;

        const auto error = eval(residuals, num_residuals, constraints, num_constraints, params_v.data());

        // std::cout << "Error : " << error << std::endl;

        if ((std::abs(last_error - error) / error) < terminate_thresold)
        {
            break;
        }

        last_error = error;
    }
    for (std::size_t i = 0; i < num_params; i++)
    {
        params[i] = params_v(i);
    }

    const auto end = std::chrono::system_clock::now();
    result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_residuals; i++)
        {
            error += (*(residuals[i]))(params);
        }
        result.final_residual_error = error;

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            error += (*constraints[i])(params);
        }
        result.final_error = error;
    }

    return result;
}

namespace optimization
{
    class function_term
    {
    public:
        virtual double eval(const double *params, std::size_t num_params) const
        {
            return 0.0;
        }
        virtual bool add_grad(const double *params, std::size_t num_params, double *nabla) const
        {
            return false;
        }
        virtual bool grad(const double *params, std::size_t num_params, double *nabla) const
        {
            return false;
        }
        virtual bool hessian(const double *params, std::size_t num_params, double *nabla, double *H) const
        {
            return false;
        }

        virtual bool add_grad(dual_t<double> *params, std::size_t num_params, double *nabla) const
        {
            return false;
        }
    };

    struct parameter_block
    {
        std::size_t offset;
        const std::size_t size;
        const std::vector<double> default_values;

        template <std::size_t size, typename... Args>
        explicit parameter_block(Args... default_values)
            : offset(std::numeric_limits<std::size_t>::max())
            , size(size)
            , default_values{std::array<double, size>{default_values...}}
        {
        }

        template <typename... Args>
        explicit parameter_block(Args... default_values)
            : offset(std::numeric_limits<std::size_t>::max())
            , size(sizeof...(Args))
            , default_values{default_values...}
        {
        }

        parameter_block(size_t size)
            : offset(std::numeric_limits<std::size_t>::max())
            , size(size)
            , default_values{}
        {
        }

        parameter_block(size_t size, const std::vector<double>& default_values)
            : offset(std::numeric_limits<std::size_t>::max())
            , size(size)
            , default_values{default_values}
        {
        }

        parameter_block()
            : offset(std::numeric_limits<std::size_t>::max())
            , size(0)
            , default_values{}
        {
        }
    };
 
    template<typename Tuple, typename Func, std::size_t... idxs>
    inline void for_each_apply(const Func &f, const Tuple &t, std::index_sequence<idxs...>)
    {
        (f(std::get<idxs>(t)), ...);
    }

    template <typename Tuple, typename Func>
    inline void for_each_apply(const Func &f, const Tuple &t)
    {
        constexpr std::size_t n = std::tuple_size_v<Tuple>;
        for_each_apply(f, t, std::make_index_sequence<n>{});
    }

    template <typename Func, typename... Params>
    class auto_diff_function_term : public function_term
    {
        using value_type = double;
        Func func;
        std::tuple<Params...> param_blocks;

        template <typename T, std::size_t... param_block_idxs>
        inline T invoke_func(const T *params, std::index_sequence<param_block_idxs...>) const
        {
            return func(&params[std::get<param_block_idxs>(param_blocks)->offset]...);
        }

        template <typename T>
        inline T invoke_func(const T *params) const
        {
            return invoke_func(params, std::index_sequence_for<Params...>());
        }

    public:
        auto_diff_function_term(const Func &func, const Params... param_blocks)
            : func(func), param_blocks(param_blocks...)
        {}

        virtual ~auto_diff_function_term() = default;

        virtual double eval(const double *params, std::size_t num_params) const override
        {
            return invoke_func(params);
        }
        virtual bool add_grad(const double *params, std::size_t num_params, double *nabla) const override
        {
            const auto v = std::make_unique<dual_t<value_type>[]>(num_params);
            for (std::size_t k = 0; k < num_params; k++)
            {
                v[k] = dual_t<value_type>(params[k], value_type{0});
            }

            const auto grad_params = [this, nabla, &v](const std::shared_ptr<parameter_block> &param_block)
            {
                if (param_block->size == 3)
                {
                    std::size_t i = param_block->offset;
                    {
                        v[i].b = value_type{1};
                        const auto value = invoke_func(v.get());
                        nabla[i] += value.b;
                        v[i].b = value_type{0};
                    }
                    {
                        v[i + 1].b = value_type{1};
                        const auto value = invoke_func(v.get());
                        nabla[i + 1] += value.b;
                        v[i + 1].b = value_type{0};
                    }
                    {
                        v[i + 2].b = value_type{1};
                        const auto value = invoke_func(v.get());
                        nabla[i + 2] += value.b;
                        v[i + 2].b = value_type{0};
                    }
                }
                else
                {
                    for (std::size_t i = param_block->offset; i < param_block->offset + param_block->size; i++)
                    {
                        v[i].b = value_type{1};

                        const auto value = invoke_func(v.get());
                        nabla[i] += value.b;

                        v[i].b = value_type{0};
                    }
                }
            };

            for_each_apply(grad_params, param_blocks);

            return true;
        }
        virtual bool grad(const double *params, std::size_t num_params, double *nabla) const override
        {
            std::fill_n(nabla, num_params, value_type{0});
            const auto v = std::make_unique<dual_t<value_type>[]>(num_params);
            for (std::size_t k = 0; k < num_params; k++)
            {
                v[k] = dual_t<value_type>(params[k], value_type{0});
            }

            const auto grad_params = [this, nabla, &v](const std::shared_ptr<parameter_block> &param_block)
            {
                for (std::size_t i = param_block->offset; i < param_block->offset + param_block->size; i++)
                {
                    v[i].b = value_type{1};

                    const auto value = invoke_func(v.get());
                    nabla[i] = value.b;

                    v[i].b = value_type{0};
                }
            };

            for_each_apply(grad_params, param_blocks);

            return true;
        }
        virtual bool hessian(const double *params, std::size_t num_params, double *nabla, double *H) const override
        {
            std::fill_n(nabla, num_params, value_type{0});
            std::fill_n(H, num_params * num_params, value_type{0});
            std::vector<hyper_dual_t<value_type>> v(num_params);

            const auto hessian1_params = [this, params, num_params, nabla, H, &v](const std::shared_ptr<parameter_block> &param_block)
            {
                for (std::size_t i = param_block->offset; i < param_block->offset + param_block->size; i++)
                {
                    for (std::size_t k = 0; k < num_params; k++)
                    {
                        v[k] = hyper_dual_t<value_type>(params[k], value_type{0}, value_type{0}, value_type{0});
                    }
                    v[i].b = value_type{1};
                    v[i].c = value_type{1};

                    const auto value = invoke_func(v.data());
                    nabla[i] = value.b;
                    H[i * num_params + i] = value.d;

                    const auto hessian2_params = [this, params, num_params, H, &v, i](const std::shared_ptr<parameter_block> &param_block)
                    {
                        for (std::size_t j = param_block->offset; j < param_block->offset + param_block->size; j++)
                        {
                            if (j <= i)
                            {
                                continue;
                            }

                            for (std::size_t k = 0; k < num_params; k++)
                            {
                                v[k] = hyper_dual_t<value_type>(params[k], value_type{0}, value_type{0}, value_type{0});
                            }
                            v[i].b = value_type{1};
                            v[j].c = value_type{1};

                            const auto value = invoke_func(v.data());

                            H[i * num_params + j] = value.d;
                            H[j * num_params + i] = value.d;
                        }
                    };

                    for_each_apply(hessian2_params, param_blocks);
                }
            };

            for_each_apply(hessian1_params, param_blocks);

            return true;
        }

        virtual bool add_grad(dual_t<value_type> *v, std::size_t num_params, double *nabla) const override
        {
            const auto grad_params = [this, nabla, &v](const std::shared_ptr<parameter_block> &param_block)
            {
                if (param_block->size == 3)
                {
                    std::size_t i = param_block->offset;
                    {
                        v[i].b = value_type{1};
                        const auto value = invoke_func(v);
                        nabla[i] += value.b;
                        v[i].b = value_type{0};
                    }
                    {
                        v[i + 1].b = value_type{1};
                        const auto value = invoke_func(v);
                        nabla[i + 1] += value.b;
                        v[i + 1].b = value_type{0};
                    }
                    {
                        v[i + 2].b = value_type{1};
                        const auto value = invoke_func(v);
                        nabla[i + 2] += value.b;
                        v[i + 2].b = value_type{0};
                    }
                }
                else
                {
                    for (std::size_t i = param_block->offset; i < param_block->offset + param_block->size; i++)
                    {
                        v[i].b = value_type{1};

                        const auto value = invoke_func(v);
                        nabla[i] += value.b;

                        v[i].b = value_type{0};
                    }
                }
            };

            for_each_apply(grad_params, param_blocks);

            return true;
        }
    };

    template <typename Func, typename... Params>
    static auto make_auto_diff_function_term(Func func, Params... params)
    {
        return std::make_shared<auto_diff_function_term<Func, Params...>>(func, params...);
    }

    class function
    {
        std::vector<std::shared_ptr<function_term>> terms;

    public:
        template <typename... Terms>
        function(std::shared_ptr<Terms>... terms)
            : terms{terms...}
        {}

        function(const std::vector<std::shared_ptr<function_term>>& terms)
            : terms(terms)
        {}
        
        double eval(const double *params, std::size_t num_params) const
        {
            auto value = 0.0;
            for (const auto &term : terms)
            {
                value += term->eval(params, num_params);
            }
            return value;
        }
        bool grad(const double *params, std::size_t num_params, double* grad) const
        {
            Eigen::Map<Eigen::VectorXd> grad_v(grad, num_params);

            const auto v = std::make_unique<dual_t<double>[]>(num_params);
            for (std::size_t k = 0; k < num_params; k++)
            {
                v[k] = dual_t<double>(params[k], double{0});
            }

            grad_v.setZero();
            for (const auto &term : terms)
            {
                if (!term->add_grad(v.get(), num_params, grad_v.data()))
                {
                    return false;
                }
            }

            return true;
        }
        bool hessian(const double *params, std::size_t num_params, double *grad, double *H) const
        {
            Eigen::Map<Eigen::MatrixXd> H_m(H, num_params, num_params);
            Eigen::MatrixXd m(num_params, num_params);

            Eigen::Map<Eigen::VectorXd> grad_v(grad, num_params);
            Eigen::VectorXd v(num_params);

            H_m.setZero();
            grad_v.setZero();
            for (const auto &term : terms)
            {
                if (!term->hessian(params, num_params, v.data(), m.data()))
                {
                    return false;
                }
                grad_v += v;
                H_m += m;
            }
            return true;
        }
    };

    static double armijo(const function &obj_func,
                         const Eigen::VectorXd &params, const Eigen::VectorXd &grad, const Eigen::VectorXd &d, std::size_t max_iteration = 10000)
    {
        auto alpha = 1.0;
        auto rho = 0.5;
        double c1 = 1e-3;

        for (std::size_t iter = 0; iter < max_iteration; iter++)
        {
            Eigen::VectorXd step_params = params + d * alpha;
            if (obj_func.eval(step_params.data(), params.size()) <=
                obj_func.eval(params.data(), params.size()) + c1 * grad.dot(d))
            {
                break;
            }
            else
            {
                alpha = alpha * rho;
            }
        }

        return alpha;
    }

    static optimization_result solve_lbfgs_method(const function& obj_func, double *params, std::size_t num_params, double terminate_thresold, std::size_t max_iteration, std::size_t max_m)
    {
        optimization_result result;

        {
            auto error = obj_func.eval(params, num_params);
            result.initial_residual_error = error;
            result.initial_error = error;
        }

        const auto start = std::chrono::system_clock::now();

        std::vector<Eigen::VectorXd> s(max_m);
        std::vector<Eigen::VectorXd> y(max_m);
        auto last_error = std::numeric_limits<double>::max();

        Eigen::VectorXd params_v(num_params);
        for (std::size_t i = 0; i < num_params; i++)
        {
            params_v(i) = params[i];
        }
        Eigen::VectorXd grad_v(num_params);
        obj_func.grad(params_v.data(), num_params, grad_v.data());

        double time1 = 0.0;

        Eigen::VectorXd next_grad_v(num_params);
        for (std::size_t iter = 0; iter < max_iteration; iter++)
        {
            Eigen::VectorXd q = -grad_v;

            if (iter > 0)
            {
                const auto m = std::min(iter, max_m);
                std::vector<double> a(max_m);

                for (std::size_t j = 0; j < m; j++)
                {
                    const auto i = iter - 1 - j;
                    a[i % max_m] = s[i % max_m].dot(q) / y[i % max_m].dot(s[i % max_m]);
                    q = q - a[i % max_m] * y[i % max_m];
                }

                const auto i = iter - 1;
                q = s[i % max_m].dot(y[i % max_m]) / y[i % max_m].dot(y[i % max_m]) * q;

                for (std::size_t j = 0; j < m; j++)
                {
                    const auto i = iter - m + j;

                    const auto b = y[i % max_m].dot(q) / y[i % max_m].dot(s[i % max_m]);
                    q = q + (a[i % max_m] - b) * s[i % max_m];
                }
            }

            Eigen::VectorXd d = q;

            for (std::size_t i = 0; i < d.size(); i++)
            {
                if (std::isnan(d(i)))
                {
                    d(i) = 0.0;
                }
            }

            const auto alpha = armijo(obj_func, params_v, grad_v, d);
            const auto step = 1.0;
            // const auto alpha = (iter == 0) ? step * 0.01 : step;

            Eigen::VectorXd next_params_v = params_v + alpha * d;

            const auto start = std::chrono::high_resolution_clock::now();
            obj_func.grad(next_params_v.data(), num_params, next_grad_v.data());

            const auto end = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0);
            // std::cout << "grad: " << elapsed << std::endl;
            time1 += elapsed;

            Eigen::VectorXd s_v = alpha * d;
            Eigen::VectorXd y_v = next_grad_v - grad_v;

            s[iter % max_m] = s_v;
            y[iter % max_m] = y_v;

            params_v = next_params_v;
            grad_v = next_grad_v;

            const auto error = obj_func.eval(params_v.data(), params_v.size());

            // std::cout << error << std::endl;
            if ((std::abs(last_error - error) / error) < terminate_thresold)
            {
                break;
            }

            last_error = error;
        }
        for (std::size_t i = 0; i < num_params; i++)
        {
            params[i] = params_v(i);
        }
        std::cout << "grad: " << time1 << std::endl;

        const auto end = std::chrono::system_clock::now();
        result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

        {
            auto error = obj_func.eval(params, num_params);
            result.final_residual_error = error;
            result.final_error = error;
        }

        return result;
    }

    static optimization_result solve_bfgs_method(const function &obj_func, double *params, std::size_t num_params, double terminate_thresold, std::size_t max_iteration)
    {
        optimization_result result;

        {
            auto error = obj_func.eval(params, num_params);
            result.initial_residual_error = error;
            result.initial_error = error;
        }

        const auto start = std::chrono::system_clock::now();

        Eigen::MatrixXd B = Eigen::MatrixXd::Identity(num_params, num_params);
        auto last_error = std::numeric_limits<double>::max();

        Eigen::VectorXd params_v(num_params);
        for (std::size_t i = 0; i < num_params; i++)
        {
            params_v(i) = params[i];
        }
        for (std::size_t iter = 0; iter < max_iteration; iter++)
        {
            Eigen::VectorXd grad_v(num_params);
            obj_func.grad(params_v.data(), num_params, grad_v.data());

            Eigen::VectorXd d = B.inverse() * -grad_v;

            const auto alpha = armijo(obj_func, params_v, grad_v, d);

            Eigen::VectorXd next_params_v = params_v + alpha * d;

            // Update B
            Eigen::VectorXd next_grad_v(num_params);
            obj_func.grad(next_params_v.data(), num_params, next_grad_v.data());

            Eigen::VectorXd s_v = alpha * d;
            Eigen::VectorXd dgrad_v = next_grad_v - grad_v;

            Eigen::VectorXd Bs_v = B * s_v;
            const auto sdgrad = s_v.dot(dgrad_v);
            const auto sBs = s_v.dot(Bs_v);

            Eigen::MatrixXd BsBs_m = Bs_v * Bs_v.transpose();
            Eigen::MatrixXd dgrad_dgrad_m = dgrad_v * dgrad_v.transpose();

            B = B - BsBs_m / sBs + dgrad_dgrad_m / sdgrad;
            params_v = next_params_v;

            const auto error = obj_func.eval(params_v.data(), num_params);

            if ((std::abs(last_error - error) / error) < terminate_thresold)
            {
                break;
            }

            last_error = error;
        }
        for (std::size_t i = 0; i < num_params; i++)
        {
            params[i] = params_v(i);
        }

        const auto end = std::chrono::system_clock::now();
        result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

        {
            auto error = obj_func.eval(params, num_params);
            result.final_residual_error = error;
            result.final_error = error;
        }

        return result;
    }

    class problem
    {
        function obj_func;
        std::vector<function> ineq_constraints;
        std::vector<function> eq_constraints;
        std::vector<std::shared_ptr<parameter_block>> param_blocks;
        std::unordered_map<std::string, double> options;
        std::vector<double> params;

        void allocate_params()
        {
            params.clear();
            for (const auto &param_block : param_blocks)
            {
                param_block->offset = params.size();
                for (std::size_t i = 0; i < param_block->size; i++)
                {
                    params.push_back(param_block->default_values[i]);
                }
            }
        }
    public:
        problem(const function& obj_func)
            : obj_func(obj_func)
        {}

        void set_option(std::string prop, double value)
        {
            options[prop] = value;
        }

        void add_ineq_constraint(const function& func)
        {
            ineq_constraints.push_back(func);
        }
        void add_eq_constraint(const function& func)
        {
            eq_constraints.push_back(func);
        }

        void add_param_block(const std::shared_ptr<parameter_block> &param_block)
        {
            param_blocks.push_back(param_block);
        }

        optimization_result solve()
        {
            allocate_params();

            {
                double terminate_thresold = 1e-7;
                std::size_t max_iteration = 50;
                std::size_t max_m = 10;
                if (const auto it = options.find("terminate_thresold"); it != options.end())
                {
                    max_iteration = static_cast<double>(it->second);
                }
                if (const auto it = options.find("max_iteration"); it != options.end())
                {
                    max_iteration = static_cast<std::size_t>(it->second);
                }
                if (const auto it = options.find("max_m"); it != options.end())
                {
                    max_iteration = static_cast<std::size_t>(it->second);
                }

                return solve_lbfgs_method(obj_func, params.data(), params.size(), terminate_thresold, max_iteration, max_m);
            }
        }

        const double* get_params(const std::shared_ptr<parameter_block> &param_block) const
        {
            if (param_block->offset == std::numeric_limits<std::size_t>::max())
            {
                throw std::runtime_error("The Parameters are not allocated");
            }
            return params.data() + param_block->offset;
        }
    };
}

template <typename ResidualFunc, typename ConstraintFunc, typename T>
static optimization_result solve_lbfgs_method(ResidualFunc **residuals, std::size_t num_residuals, ConstraintFunc **constraints, std::size_t num_constraints, T *params, std::size_t num_params, double terminate_thresold, std::size_t max_iteration, std::size_t max_m)
{
    optimization_result result;

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_residuals; i++)
        {
            error += (*(residuals[i]))(params);
        }
        result.initial_residual_error = error;

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            error += (*constraints[i])(params);
        }
        result.initial_error = error;
    }

    const auto start = std::chrono::system_clock::now();

    std::vector<Eigen::VectorXd> s(max_m);
    std::vector<Eigen::VectorXd> y(max_m);
    auto last_error = std::numeric_limits<double>::max();

    Eigen::VectorXd params_v(num_params);
    for (std::size_t i = 0; i < num_params; i++)
    {
        params_v(i) = params[i];
    }
    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        Eigen::VectorXd grad_v(num_params);
        gradient(residuals, num_residuals, constraints, num_constraints, params_v.data(), num_params, grad_v.data());

        Eigen::VectorXd q = -grad_v;

        if (iter > 0)
        {
            const auto m = std::min(iter, max_m);
            std::vector<double> a(max_m);

            for (std::size_t j = 0; j < m; j++)
            {
                const auto i = iter - 1 - j;
                a[i % max_m] = s[i % max_m].dot(q) / y[i % max_m].dot(s[i % max_m]);
                q = q - a[i % max_m] * y[i % max_m];
            }

            const auto i = iter - 1;
            q = s[i % max_m].dot(y[i % max_m]) / y[i % max_m].dot(y[i % max_m]) * q;

            for (std::size_t j = 0; j < m; j++)
            {
                const auto i = iter - m + j;

                const auto b = y[i % max_m].dot(q) / y[i % max_m].dot(s[i % max_m]);
                q = q + (a[i % max_m] - b) * s[i % max_m];
            }
        }

        Eigen::VectorXd d = q;

        const auto alpha = armijo(residuals, num_residuals, constraints, num_constraints, params_v, grad_v, d);

        Eigen::VectorXd next_params_v = params_v + alpha * d;

        Eigen::VectorXd next_grad_v(num_params);
        gradient(residuals, num_residuals, constraints, num_constraints, next_params_v.data(), num_params, next_grad_v.data());

        Eigen::VectorXd s_v = alpha * d;
        Eigen::VectorXd y_v = next_grad_v - grad_v;

        s[iter % max_m] = s_v;
        y[iter % max_m] = y_v;

        params_v = next_params_v;

        const auto error = eval(residuals, num_residuals, constraints, num_constraints, params_v.data());

        // std::cout << "Error : " << error << std::endl;

        if ((std::abs(last_error - error) / error) < terminate_thresold)
        {
            break;
        }

        last_error = error;
    }
    for (std::size_t i = 0; i < num_params; i++)
    {
        params[i] = params_v(i);
    }

    const auto end = std::chrono::system_clock::now();
    result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_residuals; i++)
        {
            error += (*(residuals[i]))(params);
        }
        result.final_residual_error = error;

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            error += (*constraints[i])(params);
        }
        result.final_error = error;
    }

    return result;
}

template <typename Func, typename T>
static optimization_result solve_root_newton_method(Func **funcs, std::size_t num_funcs, T *params, std::size_t num_params, double terminate_thresold, std::size_t max_iteration)
{
    optimization_result result;

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_funcs; i++)
        {
            error += std::abs((*(funcs[i]))(params));
        }
        result.initial_residual_error = error;
        result.initial_error = error;
    }

    const auto start = std::chrono::system_clock::now();

    std::vector<double> J(num_params * num_funcs);
    auto last_error = std::numeric_limits<double>::max();
    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        jacobi(funcs, num_funcs, params, num_params, J.data());

        Eigen::MatrixXd J_m(num_funcs, num_params);
        for (std::size_t i = 0; i < num_funcs; i++)
        {
            for (std::size_t j = 0; j < num_params; j++)
            {
                J_m(i, j) = J[i * num_params + j];
            }
        }

        Eigen::VectorXd f_v(num_funcs);
        for (std::size_t i = 0; i < num_funcs; i++)
        {
            f_v(i) = (*(funcs[i]))(params);
        }

        Eigen::VectorXd params_v(num_params);
        for (std::size_t i = 0; i < num_params; i++)
        {
            params_v(i) = params[i];
        }

        const auto next_params_v = params_v - J_m.inverse() * f_v;

        for (std::size_t i = 0; i < num_params; i++)
        {
            params[i] = next_params_v(i);
        }

        auto error = 0.0;
        for (std::size_t i = 0; i < num_funcs; i++)
        {
            error += std::abs((*(funcs[i]))(params));
        }

        std::cout << "Error : " << error << std::endl;

        if ((std::abs(last_error - error) / error) < terminate_thresold)
        {
            break;
        }

        last_error = error;
    }

    const auto end = std::chrono::system_clock::now();
    result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

    {
        auto error = 0.0;
        for (std::size_t i = 0; i < num_funcs; i++)
        {
            error += std::abs((*(funcs[i]))(params));
        }
        result.final_residual_error = error;
        result.final_error = error;
    }

    return result;
}

template <typename ResidualFunc, typename InEqConstraintFunc, typename EqConstraintFunc, typename T>
static double merit(ResidualFunc **residuals, std::size_t num_residuals,
                    InEqConstraintFunc **ineq_constraints, std::size_t num_ineq_constraints,
                    EqConstraintFunc **eq_constraints, std::size_t num_eq_constraints,
                    T *params, std::size_t num_params, double *s, double rho, double eta)
{
    T value = eval(residuals, num_residuals, params);
    value -= rho * std::accumulate(s, s + num_ineq_constraints, 0.0, [](double acc, double si) {
        return acc + std::log(si);
    });

    double sum_g = 0;
    for (std::size_t i = 0; i < num_ineq_constraints; i++)
    {
        sum_g += std::abs(eval(&ineq_constraints[i], 1, params) + s[i]);
    }
    value += eta * sum_g;

    double sum_h = 0;
    for (std::size_t i = 0; i < num_eq_constraints; i++)
    {
        sum_h += std::abs(eval(&eq_constraints[i], 1, params));
    }
    value += eta * sum_h;

    return value;
}

template <typename ResidualFunc, typename InEqConstraintFunc, typename EqConstraintFunc, typename T>
static double backtracking_line_search(ResidualFunc **residuals, std::size_t num_residuals,
                                     InEqConstraintFunc **ineq_constraints, std::size_t num_ineq_constraints,
                                     EqConstraintFunc **eq_constraints, std::size_t num_eq_constraints,
                                     T *params, std::size_t num_params,
                                     double* s, double* u,  T *dparams, double* ds, double* du, double rho, double eta, double beta)
{
    double alpha1 = 1.0;
    double alpha2 = 1.0;

    double min_sds = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < num_ineq_constraints; i++)
    {
        if (ds[i] < 0.0)
        {
            min_sds = std::min(min_sds, -s[i] / ds[i]);
        }
    }

    if (min_sds != std::numeric_limits<double>::max())
    {
        alpha1 = min_sds;
    }

    double min_udu = std::numeric_limits<double>::max();
    for (std::size_t i = 0; i < num_ineq_constraints; i++)
    {
        if (du[i] < 0.0)
        {
            min_udu = std::min(min_udu, -u[i] / du[i]);
        }
    }

    if (min_udu != std::numeric_limits<double>::max())
    {
        alpha2 = min_udu;
    }

    double alpha = std::min(1.0, std::min(alpha1 * beta, alpha2 * beta));

    double min_merit = std::numeric_limits<double>::max();
    double final_alpha = alpha;
    double initial_alpha = alpha;
    
    Eigen::VectorXd params_v = Eigen::Map<Eigen::VectorXd>(params, num_params);
    Eigen::VectorXd dparams_v = Eigen::Map<Eigen::VectorXd>(dparams, num_params);
    Eigen::VectorXd s_v = Eigen::Map<Eigen::VectorXd>(s, num_ineq_constraints);
    Eigen::VectorXd ds_v = Eigen::Map<Eigen::VectorXd>(ds, num_ineq_constraints);

    while (alpha > initial_alpha * 0.1)
    {
        Eigen::VectorXd new_params_v = params_v + alpha * dparams_v;
        Eigen::VectorXd new_s_v = s_v + alpha * ds_v;

        double m = merit(residuals, num_residuals, ineq_constraints, num_ineq_constraints,
            eq_constraints, num_eq_constraints, new_params_v.data(), num_params, new_s_v.data(), rho, eta);

        if (m < min_merit)
        {
            min_merit = m;
            final_alpha = alpha;
        }

        alpha *= beta;
    }

    return final_alpha;
}

template <typename ResidualFunc, typename InEqConstraintFunc, typename EqConstraintFunc, typename T>
static optimization_result solve_interior_point_method(ResidualFunc **residuals, std::size_t num_residuals,
                                                       InEqConstraintFunc **ineq_constraints, std::size_t num_ineq_constraints,
                                                       EqConstraintFunc **eq_constraints, std::size_t num_eq_constraints,
                                                       T *params, std::size_t num_params, double terminate_thresold, std::size_t max_iteration, double eta=0.1, double beta=0.9, double t=0.5)
{
    optimization_result result;

    const auto start = std::chrono::system_clock::now();

    Eigen::VectorXd s_v = Eigen::VectorXd::Ones(num_ineq_constraints);
    double rho = 1.0;
    Eigen::VectorXd u_v = rho / s_v.array();
    Eigen::VectorXd v_v = Eigen::VectorXd::Zero(num_eq_constraints);

    Eigen::VectorXd params_v = Eigen::VectorXd::Ones(num_params);
    // Eigen::VectorXd params_v(num_params);
    // for (std::size_t i = 0; i < num_params; i++)
    // {
    //     params_v(i) = params[i];
    // }

    auto last_error = std::numeric_limits<double>::max();
    for (std::size_t iter = 0; iter < max_iteration; iter++)
    {
        Eigen::MatrixXd H_m(num_params, num_params);
        Eigen::VectorXd d_v(num_params);
        hessian(residuals, num_residuals, params_v.data(), num_params, d_v.data(), H_m.data());

        Eigen::MatrixXd Jg_m(num_ineq_constraints, num_params);
        Eigen::VectorXd g_v(num_ineq_constraints);

        for (std::size_t i = 0; i < num_ineq_constraints; i++)
        {
            Eigen::MatrixXd H_m2(num_params, num_params);
            Eigen::VectorXd d_v2(num_params);
            hessian(&ineq_constraints[i], 1, params_v.data(), num_params, d_v2.data(), H_m2.data());

            H_m += H_m2 * u_v(i);
            Jg_m.row(i) = d_v2;
            d_v += d_v2 * u_v(i);
            g_v(i) = eval(&ineq_constraints[i], 1, params_v.data());
        }

        Eigen::MatrixXd Jh_m(num_eq_constraints, num_params);
        Eigen::VectorXd h_v(num_eq_constraints);

        for (std::size_t i = 0; i < num_eq_constraints; i++)
        {
            Eigen::MatrixXd H_m2(num_params, num_params);
            Eigen::VectorXd d_v2(num_params);
            hessian(&eq_constraints[i], 1, params_v.data(), num_params, d_v2.data(), H_m2.data());

            H_m += H_m2 * v_v(i);
            Jh_m.row(i) = d_v2;
            d_v += d_v2 * v_v(i);
            h_v(i) = eval(&eq_constraints[i], 1, params_v.data());
        }

        Eigen::MatrixXd Du_m = u_v.asDiagonal();
        Eigen::MatrixXd Ds_m = s_v.asDiagonal();

        Eigen::MatrixXd A_m = Eigen::MatrixXd::Zero(num_params + num_ineq_constraints + num_ineq_constraints + num_eq_constraints,
                                                    num_params + num_ineq_constraints + num_ineq_constraints + num_eq_constraints);

        Eigen::MatrixXd Jg_m_T = Jg_m.transpose();
        Eigen::MatrixXd Jh_m_T = Jh_m.transpose();

        // |H   O   Jg.T  Jh.T|
        // |O   Du  Ds    O   |
        // |Jg  I   O     O   |
        // |Jh  O   O     O   |
        A_m.block(0, 0, H_m.rows(), H_m.cols()) = H_m;
        A_m.block(0, num_params + num_ineq_constraints, Jg_m_T.rows(), Jg_m_T.cols()) = Jg_m_T;
        A_m.block(0, num_params + 2 * num_ineq_constraints, Jh_m_T.rows(), Jh_m_T.cols()) = Jh_m_T;
        A_m.block(num_params, num_params, Du_m.rows(), Du_m.cols()) = Du_m;
        A_m.block(num_params, num_params + num_ineq_constraints, Ds_m.rows(), Ds_m.cols()) = Ds_m;
        A_m.block(num_params + num_ineq_constraints, 0, Jg_m.rows(), Jg_m.cols()) = Jg_m;
        A_m.block(num_params + num_ineq_constraints, num_params, num_ineq_constraints, num_ineq_constraints)
            = Eigen::MatrixXd::Identity(num_ineq_constraints, num_ineq_constraints);
        A_m.block(num_params + 2 * num_ineq_constraints, 0, Jh_m.rows(), Jh_m.cols()) = Jh_m;

        // |-d, rho - s * u, -(g + s), -h|
        Eigen::VectorXd b0_v = -d_v;
        Eigen::VectorXd b1_v = rho - s_v.array() * u_v.array();
        Eigen::VectorXd b2_v = -(g_v + s_v);
        Eigen::VectorXd b3_v = -h_v;

        Eigen::VectorXd b_v(b0_v.size() + b1_v.size() + b2_v.size() + b3_v.size());
        b_v << b0_v, b1_v, b2_v, b3_v;

        Eigen::VectorXd delta_v = A_m.householderQr().solve(b_v);

        Eigen::VectorXd dparams_v = delta_v.segment(0, num_params);
        Eigen::VectorXd ds_v = delta_v.segment(num_params, num_ineq_constraints);
        Eigen::VectorXd du_v = delta_v.segment(num_params + num_ineq_constraints, num_ineq_constraints);
        Eigen::VectorXd dv_v = delta_v.segment(num_params + 2 * num_ineq_constraints, num_eq_constraints);

        double alpha = backtracking_line_search(residuals, num_residuals,
                                                ineq_constraints, num_ineq_constraints,
                                                eq_constraints, num_eq_constraints,
                                                params_v.data(), num_params,
                                                s_v.data(), u_v.data(), dparams_v.data(), ds_v.data(), du_v.data(), rho, eta, beta);

        params_v += alpha * dparams_v;
        s_v += alpha * ds_v;
        u_v += alpha * du_v;
        v_v += alpha * dv_v;

        if (num_ineq_constraints)
        {
            rho = t * u_v.dot(s_v) / num_ineq_constraints;
        }
    }
    for (std::size_t i = 0; i < num_params; i++)
    {
        params[i] = params_v(i);
    }

    const auto end = std::chrono::system_clock::now();
    result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

    return result;
}
