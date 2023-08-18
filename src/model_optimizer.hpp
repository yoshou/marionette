#pragma once

#include <cstddef>
#include <vector>
#include <random>

#include <glm/glm.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "model.hpp"
#include "camera_info.hpp"
#include "frame.hpp"
#include "registration.hpp"
#include "transform.hpp"
#include "nearest_neighbor.hpp"
#include "object_functions.hpp"
#include "debug.hpp"

#include <iostream>

struct zero_function
{
    template <typename T>
    T operator()(const T *params) const
    {
        return T(0.0);
    }
};

#if 0
class model_optimizer
{
    using residual_functor = srt_least_square_functor;
    using constraint_functor = srt_articulation_least_square_constraint;

    const model_data* model;

    std::map<std::string, std::shared_ptr<parameter>> scale_params;
    std::vector<std::weak_ptr<parameter>> parameters;

    struct articulation_equation
    {
        std::shared_ptr<residual_functor> residual1;
        std::shared_ptr<residual_functor> residual2;
        std::shared_ptr<constraint_functor> constraint;
    };

    std::map<std::size_t, std::vector<articulation_equation>> equations;

    void remove_unused_parameters()
    {
        const auto iter = std::remove_if(parameters.begin(), parameters.end(), [](const std::weak_ptr<parameter>& parameter) {
            return parameter.expired();
        });
        parameters.erase(iter, parameters.end());
    }
public:
    std::size_t frame_no;

    model_optimizer(const model_data &model)
        : model(&model)
    {
        for (const auto &p : model.clusters)
        {
            const auto scale_param = std::make_shared<parameter>(3, 1.0);
            scale_params.insert(std::make_pair(p.first, scale_param));
            parameters.push_back(scale_param);
        }
    }

    void update(const frame_data_t &keyframe, model_instance_data &model_instance)
    {
        auto &registered_clusters = model_instance.clusters;
        std::random_device rd;
        std::mt19937 gen(rd());

        for (auto &equation : equations)
        {
            std::shuffle(equation.second.begin(), equation.second.end(), gen);
            equation.second.resize(std::min(equation.second.size(), static_cast<std::size_t>(3)));
        }

        remove_unused_parameters();

        std::cout << parameters.size() << std::endl;
        std::vector<glm::vec3> target_points;
        std::vector<glm::vec3> points;
        std::vector<int> cluster_index;
        std::vector<float> twist_weights;
        for (std::size_t i = 0; i < registered_clusters.size(); i++)
        {
            const auto &registered_cluster = registered_clusters[i];

            std::vector<glm::vec3> source_points;
            transform_to_target(registered_cluster.cluster, registered_cluster.fit_result, source_points);
            const auto markers = keyframe.markers;

            const auto cluster_pose = transform_to_target_pose(registered_cluster.cluster, registered_cluster.fit_result);

            const auto twist_axis = to_line(cluster_pose);

            std::vector<float> dists(markers.size());
            std::fill(dists.begin(), dists.end(), std::numeric_limits<float>::max());
            for (std::size_t j = 0; j < source_points.size(); j++)
            {
                const auto point = source_points[j];
                const auto twist_weight = registered_cluster.cluster.points[j].weight;
                const auto closest_point = find_nearest_neighbor_point(point, markers, dists);

                points.push_back(source_points[j]);
                twist_weights.push_back(twist_weight);
                target_points.push_back(closest_point);
                cluster_index.push_back(i);
            }
        }

        std::vector<std::vector<glm::vec3>> points_s(registered_clusters.size());
        std::vector<std::vector<glm::vec3>> points_t(registered_clusters.size());
        std::vector<glm::vec3> origins(registered_clusters.size());

        for (std::size_t i = 0; i < points.size(); ++i)
        {
            const auto point = points[i];
            const auto target_point = target_points[i];

            points_s[cluster_index[i]].push_back(point);
            points_t[cluster_index[i]].push_back(target_point);
        }

        for (std::size_t i = 0; i < registered_clusters.size(); i++)
        {
            const auto origin = glm::vec3(transform_to_target_pose(registered_clusters[i].cluster, registered_clusters[i].fit_result)[3]);
            origins[i] = origin;
        }

        std::size_t num_clusters = registered_clusters.size();

        const auto articulation_pairs = get_articulation_pairs();
        const auto num_constraints = articulation_pairs.size();

        std::vector<std::shared_ptr<residual_functor>> residuals;
        std::vector<std::shared_ptr<constraint_functor>> constraints;
        std::vector<constraint_functor *> constraint_ptrs;

        std::map<std::string, std::shared_ptr<parameter>> rotate_params;
        std::map<std::string, std::shared_ptr<parameter>> translate_params;

        for (std::size_t i = 0; i < num_clusters; i++)
        {
            const auto cluster_name = registered_clusters[i].cluster.name;

            const auto scale_param = scale_params.at(cluster_name);

            const auto rotate_param = std::make_shared<parameter>(3);
            rotate_params.insert(std::make_pair(cluster_name, rotate_param));

            const auto translate_param = std::make_shared<parameter>(3);
            translate_params.insert(std::make_pair(cluster_name, translate_param));

            parameters.push_back(rotate_param);
            parameters.push_back(translate_param);

            const auto residual = std::make_shared<residual_functor>(points_s[i], points_t[i], origins[i]);
            residual->parameters.push_back(scale_param);
            residual->parameters.push_back(rotate_param);
            residual->parameters.push_back(translate_param);

            residuals.push_back(residual);
        }
        std::vector<residual_functor *> residual_ptrs;
        for (const auto& residual : residuals)
        {
            residual_ptrs.push_back(residual.get());
        }

        std::map<std::string, std::size_t> cluster_index_map;
        for (std::size_t j = 0; j < registered_clusters.size(); j++)
        {
            const auto &cluster = registered_clusters[j];
            cluster_index_map.insert(std::make_pair(cluster.cluster.name, j));
        }
        for (std::size_t i = 0; i < num_constraints; i++)
        {
            const auto [parent_name, child_name] = articulation_pairs[i];

            const auto parent_cluster_idx = cluster_index_map.at(parent_name);
            const auto child_cluster_idx = cluster_index_map.at(child_name);
            const auto &parent_cluster = registered_clusters[parent_cluster_idx];
            const auto &child_cluster = registered_clusters[child_cluster_idx];
            const auto [pos1, rot1, pos2, rot2] = compute_articuated_pair_points(parent_cluster.cluster, child_cluster.cluster, parent_cluster.fit_result, child_cluster.fit_result);

            const auto constraint = std::make_shared<constraint_functor>(pos1, pos2, origins[parent_cluster_idx], origins[child_cluster_idx]);

            const auto parent_scale_param = scale_params.at(parent_name);
            const auto parent_rotate_param = rotate_params.at(parent_name);
            const auto parent_translate_param = translate_params.at(parent_name);
            const auto child_scale_param = scale_params.at(child_name);
            const auto child_rotate_param = rotate_params.at(child_name);
            const auto child_translate_param = translate_params.at(child_name);

            constraint->parameters.push_back(parent_scale_param);
            constraint->parameters.push_back(parent_rotate_param);
            constraint->parameters.push_back(parent_translate_param);
            constraint->parameters.push_back(child_scale_param);
            constraint->parameters.push_back(child_rotate_param);
            constraint->parameters.push_back(child_translate_param);

            constraints.push_back(constraint);
            constraint_ptrs.push_back(constraint.get());

            const auto equation = articulation_equation{
                residuals[parent_cluster_idx],
                residuals[child_cluster_idx],
                constraint};
            equations[i].push_back(equation);
        }

        std::vector<double> params;
        for (const auto &param : parameters)
        {
            if (const auto ptr = param.lock())
            {
                ptr->offset = params.size();
                for (std::size_t i = 0; i < ptr->num; i++)
                {
                    params.push_back(ptr->initial_value);
                }
            }
        }
        const auto num_params = params.size();

        std::vector<zero_function*> ineq_funcs;

        // const auto opt_result = solve_newton_method(residual_ptrs.data(), residual_ptrs.size(), constraint_ptrs.data(), constraint_ptrs.size(), params.data(), num_params, 1e-5, 100);
        // const auto opt_result = solve_semi_newton_method(residual_ptrs.data(), residual_ptrs.size(), constraint_ptrs.data(), constraint_ptrs.size(), params.data(), num_params, 1e-5, 100);
        const auto opt_result = solve_lbfgs_method(residual_ptrs.data(), residual_ptrs.size(), constraint_ptrs.data(), constraint_ptrs.size(), params.data(), num_params, 1e-7, 100, 5);
        // const auto opt_result = solve_interior_point_method(residual_ptrs.data(), residual_ptrs.size(), ineq_funcs.data(), ineq_funcs.size(), constraint_ptrs.data(), constraint_ptrs.size(), params.data(), num_params, 1e-7, 100);

        // std::cout << "Initial error : " << opt_result.initial_error << std::endl;
        // std::cout << "Initial residual error : " << opt_result.initial_residual_error << std::endl;
        // std::cout << "Final error : " << opt_result.final_error << std::endl;
        // std::cout << "Final residual error : " << opt_result.final_residual_error << std::endl;
        // std::cout << "Total elapsed time : " << opt_result.total_elapsed_time << std::endl;

#if 1
    {
        point_cloud_debug_drawer debug_point;
        for (std::size_t i = 0; i < num_clusters; i++)
        {
            const auto& src_points = points_s[i];
            const auto& target_points = points_t[i];
            const auto origin = origins[i];

            for (std::size_t j = 0; j < src_points.size(); j++)
            {
                // debug_point.add(src_points[i], glm::u8vec3(0, 0, 255));
                debug_point.add(src_points[j], glm::u8vec3(0, 0, 255));
                debug_point.add(target_points[j], glm::u8vec3(255, 0, 0));
            }
        }

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            const auto [parent_name, child_name] = articulation_pairs[i];

            const auto parent_cluster_idx = cluster_index_map.at(parent_name);
            const auto child_cluster_idx = cluster_index_map.at(child_name);
            const auto &parent_cluster = registered_clusters[parent_cluster_idx];
            const auto &child_cluster = registered_clusters[child_cluster_idx];
            const auto [pos1, rot1, pos2, rot2] = compute_articuated_pair_points(parent_cluster.cluster, child_cluster.cluster, parent_cluster.fit_result, child_cluster.fit_result);

            debug_point.add(pos1, glm::u8vec3(0, 255, 255));
            debug_point.add(pos2, glm::u8vec3(255, 255, 0));
        }

        debug_point.save("./debug_cloud_src_target" + std::to_string(frame_no) + ".pcd");
    }
    {
        point_cloud_debug_drawer debug_point;
        for (std::size_t i = 0; i < num_clusters; i++)
        {
            auto &registered_cluster = registered_clusters[i];

            const auto scale = params.data() + scale_params.at(registered_cluster.cluster.name)->offset;
            const auto rotation = params.data() + rotate_params.at(registered_cluster.cluster.name)->offset;
            const auto translation = params.data() + translate_params.at(registered_cluster.cluster.name)->offset;

            const auto &src_points = points_s[i];
            const auto &target_points = points_t[i];
            const auto origin = origins[i];

            for (std::size_t j = 0; j < src_points.size(); j++)
            {
                double point[3] = {
                    static_cast<double>(src_points[j].x - origin.x),
                    static_cast<double>(src_points[j].y - origin.y),
                    static_cast<double>(src_points[j].z - origin.z)};

                point[0] *= scale[0];
                point[1] *= scale[1];
                point[2] *= scale[2];

                double p[3];
                ceres::AngleAxisRotatePoint(rotation, point, p);

                p[0] += static_cast<double>(origin.x);
                p[1] += static_cast<double>(origin.y);
                p[2] += static_cast<double>(origin.z);

                // if (std::is_same_v<residual_functor, srt_least_square_functor>)
                {
                    p[0] += translation[0];
                    p[1] += translation[1];
                    p[2] += translation[2];
                }

                const glm::vec3 transformed_point(
                    static_cast<float>(p[0]),
                    static_cast<float>(p[1]),
                    static_cast<float>(p[2]));

                debug_point.add(transformed_point, glm::u8vec3(0, 0, 255));
                debug_point.add(target_points[j], glm::u8vec3(255, 0, 0));
            }
        }

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            const auto [parent_name, child_name] = articulation_pairs[i];

            const auto parent_cluster_idx = cluster_index_map.at(parent_name);
            const auto child_cluster_idx = cluster_index_map.at(child_name);
            const auto &parent_cluster = registered_clusters[parent_cluster_idx];
            const auto &child_cluster = registered_clusters[child_cluster_idx];
            const auto [pos1, rot1, pos2, rot2] = compute_articuated_pair_points(parent_cluster.cluster, child_cluster.cluster, parent_cluster.fit_result, child_cluster.fit_result);

            const auto parent_origin = glm::vec3(parent_cluster.cluster.pose[3]);
            const auto child_origin = glm::vec3(child_cluster.cluster.pose[3]);

            const auto parent_scale = params.data() + scale_params.at(parent_name)->offset;
            const auto parent_rotation = params.data() + rotate_params.at(parent_name)->offset;
            const auto parent_translation = params.data() + translate_params.at(parent_name)->offset;
            const auto child_scale = params.data() + scale_params.at(child_name)->offset;
            const auto child_rotation = params.data() + rotate_params.at(child_name)->offset;
            const auto child_translation = params.data() + translate_params.at(child_name)->offset;

            double point1[3] = {
                static_cast<double>(pos1.x - parent_origin.x),
                static_cast<double>(pos1.y - parent_origin.y),
                static_cast<double>(pos1.z - parent_origin.z)};
            double point2[3] = {
                static_cast<double>(pos2.x - child_origin.x),
                static_cast<double>(pos2.y - child_origin.y),
                static_cast<double>(pos2.z - child_origin.z)};

            point1[0] *= parent_scale[0];
            point1[1] *= parent_scale[1];
            point1[2] *= parent_scale[2];

            point2[0] *= child_scale[0];
            point2[1] *= child_scale[1];
            point2[2] *= child_scale[2];

            // Transform articulation by parent
            double p1[3];
            ceres::AngleAxisRotatePoint(parent_rotation, point1, p1);

            p1[0] += static_cast<double>(parent_origin.x);
            p1[1] += static_cast<double>(parent_origin.y);
            p1[2] += static_cast<double>(parent_origin.z);

            // if (std::is_same_v<residual_functor, srt_least_square_functor>)
            {
                p1[0] += parent_translation[0];
                p1[1] += parent_translation[1];
                p1[2] += parent_translation[2];
            }

            // Transform articulation by child
            double p2[3];
            ceres::AngleAxisRotatePoint(child_rotation, point2, p2);

            p2[0] += static_cast<double>(child_origin.x);
            p2[1] += static_cast<double>(child_origin.y);
            p2[2] += static_cast<double>(child_origin.z);

            // if (std::is_same_v<residual_functor, srt_least_square_functor>)
            {
                p2[0] += child_translation[0];
                p2[1] += child_translation[1];
                p2[2] += child_translation[2];
            }

            const glm::vec3 transformed_point1(
                static_cast<float>(p1[0]),
                static_cast<float>(p1[1]),
                static_cast<float>(p1[2]));

            const glm::vec3 transformed_point2(
                static_cast<float>(p2[0]),
                static_cast<float>(p2[1]),
                static_cast<float>(p2[2]));

            debug_point.add(transformed_point1 + glm::vec3(0.01f, 0.f, 0.f), glm::u8vec3(0, 255, 255));
            debug_point.add(transformed_point2, glm::u8vec3(255, 255, 0));
        }

        debug_point.save("./debug_cloud_src_transform" + std::to_string(frame_no) + ".pcd");
    }
#endif

        for (std::size_t i = 0; i < num_clusters; i++)
        {
            auto &registered_cluster = registered_clusters[i];

            const auto scale = params.data() + scale_params.at(registered_cluster.cluster.name)->offset;
            const auto rotation = params.data() + rotate_params.at(registered_cluster.cluster.name)->offset;
            const auto translation = params.data() + translate_params.at(registered_cluster.cluster.name)->offset;

            const auto cluster_pose = transform_to_target_pose(registered_cluster.cluster, registered_cluster.fit_result);

            const auto transform_by_param = [](glm::mat4 m, glm::vec3 origin, const double *scale, const double *axis_angle, const double *translation)
            {
                glm::vec3 scale_vec(static_cast<float>(scale[0]),
                                    static_cast<float>(scale[1]),
                                    static_cast<float>(scale[2]));

                glm::vec3 axis_angle_vec(static_cast<float>(axis_angle[0]),
                                         static_cast<float>(axis_angle[1]),
                                         static_cast<float>(axis_angle[2]));

                glm::vec3 trans_vec(static_cast<float>(translation[0]),
                                    static_cast<float>(translation[1]),
                                    static_cast<float>(translation[2]));

                const auto angle = glm::length(axis_angle_vec);

                if (angle > std::numeric_limits<float>::epsilon())
                {
                    const auto axis = glm::normalize(axis_angle_vec);
                    const auto quat = glm::angleAxis(angle, axis);

                    return glm::translate(trans_vec) * glm::translate(origin) * glm::toMat4(quat) * glm::scale(scale_vec) * glm::translate(-origin) * m;
                }
                else
                {
                    return glm::translate(trans_vec) * glm::translate(origin) * glm::scale(scale_vec) * glm::translate(-origin) * m;
                }
            };

            const auto origin = origins[i];

            const auto transformed_pose = transform_by_param(cluster_pose, origin, scale, rotation, translation);

            // registered_cluster.cluster.pose = transform_to_source_pose(transformed_pose, registered_cluster.cluster, registered_cluster.fit_result);

            auto trans = glm::translate(registered_cluster.fit_result.translation) * glm::mat4(registered_cluster.fit_result.rotation);
            trans = transform_by_param(trans, origin, scale, rotation, translation);

            registered_cluster.fit_result.rotation = glm::mat3(trans);
            registered_cluster.fit_result.translation = glm::vec3(trans[3]);
        }

#if 1
    {
        point_cloud_debug_drawer debug_point;
        for (std::size_t i = 0; i < num_clusters; i++)
        {
            const auto &src_points = points_s[i];
            const auto &target_points = points_t[i];
            const auto origin = origins[i];

            for (std::size_t j = 0; j < src_points.size(); j++)
            {
                // debug_point.add(src_points[i], glm::u8vec3(0, 0, 255));
                debug_point.add(src_points[j], glm::u8vec3(0, 0, 255));
                debug_point.add(target_points[j], glm::u8vec3(255, 0, 0));
            }
        }

        for (std::size_t i = 0; i < num_constraints; i++)
        {
            const auto [parent_name, child_name] = articulation_pairs[i];

            const auto parent_cluster_idx = cluster_index_map.at(parent_name);
            const auto child_cluster_idx = cluster_index_map.at(child_name);
            const auto &parent_cluster = registered_clusters[parent_cluster_idx];
            const auto &child_cluster = registered_clusters[child_cluster_idx];
            const auto [pos1, rot1, pos2, rot2] = compute_articuated_pair_points(parent_cluster.cluster, child_cluster.cluster, parent_cluster.fit_result, child_cluster.fit_result);

            debug_point.add(pos1, glm::u8vec3(0, 255, 255));
            debug_point.add(pos2, glm::u8vec3(255, 255, 0));
        }

        debug_point.save("./debug_cloud_src_target_optimized" + std::to_string(frame_no) + ".pcd");
    }
#endif
    }

    void reset(const std::string& cluster_name)
    {
    }
};
#endif
