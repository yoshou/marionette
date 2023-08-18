#pragma once

#include <chrono>
#include <numeric>
#include <memory>
#include <cstdint>

#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/QR>

#include "nonlinear_solver.hpp"
#include "automatic_differentiation.hpp"

#include "unsupported/Eigen/NonLinearOptimization"

namespace optimization
{
    class residual_block
    {
    public:
        virtual bool eval(const double *params, std::size_t num_params, double *values) const
        {
            return false;
        }
        virtual bool jacobian(const double *params, std::size_t num_params, double *jac) const
        {
            return false;
        }
        virtual std::size_t num_values() const
        {
            return 0;
        }
    };

    template <std::size_t num_values_, typename Func, typename... Params>
    class auto_diff_residual_block : public residual_block
    {
        using value_type = double;
        Func func;
        std::tuple<Params...> param_blocks;

        template <typename T, std::size_t... param_block_idxs>
        inline bool invoke_func(const T *params, T *values, std::index_sequence<param_block_idxs...>) const
        {
            return func(&params[std::get<param_block_idxs>(param_blocks)->offset]..., values);
        }

        template <typename T>
        inline bool invoke_func(const T *params, T *values) const
        {
            return invoke_func(params, values, std::index_sequence_for<Params...>());
        }

    public:
        auto_diff_residual_block(const Func &func, const Params... param_blocks)
            : func(func), param_blocks(param_blocks...)
        {}

        virtual ~auto_diff_residual_block() = default;

        virtual std::size_t num_values() const override
        {
            return num_values_;
        }

        virtual bool eval(const double *params, std::size_t num_params, double *values) const override
        {
            return invoke_func(params, values);
        }
        virtual bool jacobian(const double *params, std::size_t num_params, double *jac) const override
        {
            const auto v = std::make_unique<dual_t<value_type>[]>(num_params);
            for (std::size_t k = 0; k < num_params; k++)
            {
                v[k] = dual_t<value_type>(params[k], value_type{0});
            }

            const auto grad_params = [this, num_params, jac, &v](const std::shared_ptr<parameter_block> &param_block)
            {
                for (std::size_t i = param_block->offset; i < param_block->offset + param_block->size; i++)
                {
                    v[i].b = value_type{1};

                    std::array<dual_t<value_type>, num_values_> values;
                    if (!invoke_func(v.get(), values.data()))
                    {
                        return false;
                    }

                    for (std::size_t j = 0; j < values.size(); j++)
                    {
                        jac[num_params * j + i] = values[j].b;
                    }

                    v[i].b = value_type{0};
                }

                return true;
            };

            for_each_apply(grad_params, param_blocks);

            return true;
        }
    };

    template <std::size_t num_values, typename Func, typename... Params>
    static auto make_auto_diff_residual_block(Func func, Params... params)
    {
        return std::make_shared<auto_diff_residual_block<num_values, Func, Params...>>(func, params...);
    }

    class residual_vector
    {
        std::vector<std::shared_ptr<residual_block>> residuals;

    public:
        template <typename... Residuals>
        residual_vector(std::shared_ptr<Residuals>... residuals)
            : residuals{residuals...}
        {}

        residual_vector(const std::vector<std::shared_ptr<residual_block>> &residuals)
            : residuals(residuals)
        {}

        std::size_t num_values() const
        {
            std::size_t count = 0;
            for (const auto &residual : residuals)
            {
                count += residual->num_values();
            }
            return count;
        }

        bool eval(const double *params, std::size_t num_params, double* values) const
        {
            for (const auto &residual : residuals)
            {
                if (!residual->eval(params, num_params, values))
                {
                    return false;
                }
                values += residual->num_values();
            }
            return true;
        }
        bool jacobian(const double *params, std::size_t num_params, double* jac) const
        {
            for (const auto &residual : residuals)
            {
                if (!residual->jacobian(params, num_params, jac))
                {
                    return false;
                }
                jac += (num_params * residual->num_values());
            }

            return true;
        }
        double squared(const double *params, std::size_t num_params) const
        {
            Eigen::VectorXd values_v(num_values());

            double* values = values_v.data();
            for (const auto &residual : residuals)
            {
                if (!residual->eval(params, num_params, values))
                {
                    return false;
                }
                values += residual->num_values();
            }
            return values_v.squaredNorm();
        }
    };

    class least_square_problem
    {
        residual_vector residuals;
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
        least_square_problem(const residual_vector &residuals)
            : residuals(residuals)
        {
        }

        void set_option(std::string prop, double value)
        {
            options[prop] = value;
        }

        void add_param_block(const std::shared_ptr<parameter_block> &param_block)
        {
            param_blocks.push_back(param_block);
        }

        struct LevenbergMarquardtFunc
        {
            residual_vector func;
            const std::size_t num_params;
            const std::size_t num_values;

            LevenbergMarquardtFunc(const residual_vector &func, std::size_t num_params)
                : func(func), num_params(num_params), num_values(func.num_values()) {}

            int operator()(const Eigen::VectorXd &b, Eigen::VectorXd &fvec) const
            {
                func.eval(b.data(), b.size(), fvec.data());
                return 0;
            }
            int df(const Eigen::VectorXd &b, Eigen::MatrixXd &fjac)
            {
                Eigen::MatrixXd jac(fjac.cols(), fjac.rows());
                jac.setZero();
                func.jacobian(b.data(), b.size(), jac.data());
                fjac = jac.transpose();
                return 0;
            }

            int inputs() const { return static_cast<int>(num_params); }
            int values() const { return static_cast<int>(num_values); }
        };

        optimization_result solve()
        {
            allocate_params();

            {
                double terminate_thresold = 1e-7;
                std::size_t max_iteration = 50;
                if (const auto it = options.find("terminate_thresold"); it != options.end())
                {
                    max_iteration = static_cast<double>(it->second);
                }
                if (const auto it = options.find("max_iteration"); it != options.end())
                {
                    max_iteration = static_cast<std::size_t>(it->second);
                }

                LevenbergMarquardtFunc functor(residuals, params.size());
                Eigen::LevenbergMarquardt<LevenbergMarquardtFunc> lm(functor);

                Eigen::VectorXd params_v(params.size());
                for (std::size_t i = 0; i < params.size(); i++)
                {
                    params_v(i) = params[i];
                }

                optimization_result result;

                {
                    const auto error = residuals.squared(params_v.data(), params_v.size());
                    result.initial_residual_error = error;
                    result.initial_error = error;
                }
                lm.parameters.maxfev = max_iteration;
                lm.parameters.xtol = terminate_thresold;
                const auto start = std::chrono::system_clock::now();
                const auto info = lm.minimize(params_v);
                const auto end = std::chrono::system_clock::now();
                for (std::size_t i = 0; i < params.size(); i++)
                {
                    params[i] = params_v(i);
                }

                result.total_elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / (1000.0 * 1000.0);

                {
                    auto error = residuals.squared(params_v.data(), params_v.size());
                    result.final_residual_error = error;
                    result.final_error = error;
                }

                return result;
            }
        }

        const double *get_params(const std::shared_ptr<parameter_block> &param_block) const
        {
            if (param_block->offset == std::numeric_limits<std::size_t>::max())
            {
                throw std::runtime_error("The Parameters are not allocated");
            }
            return params.data() + param_block->offset;
        }
    };
}
