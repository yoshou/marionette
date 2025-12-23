#include <cstddef>
#include <vector>

#include <glm/glm.hpp>
#include <glm/gtx/string_cast.hpp>

#include "model.hpp"
#include "camera_info.hpp"
#include "frame.hpp"
#include "registration.hpp"
#include "transform.hpp"
#include "nearest_neighbor.hpp"
#include "debug.hpp"
#include "nonlinear_solver.hpp"
#include "nonlinear_least_square_solver.hpp"
#include "object_functions.hpp"
#include "srt_transform.hpp"
#include "model_detector.hpp"
#include "articulation_solver.hpp"

#include <iostream>

#include "icp_3d_3d_optimizer.hpp"

#define USE_CERES_SOLVER 1

#if USE_CERES_SOLVER
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#endif

class closest_point_correnspondance_matcher final : public correnspondance_matcher
{
public:
    virtual std::vector<std::vector<std::pair<std::pair<std::size_t, std::size_t>, float>>> find_correnspondance(
        const std::vector<std::vector<weighted_point>> &sources,
        const std::vector<std::vector<weighted_point>> &search_points,
        const point_cloud &target) override
    {
        std::vector<std::vector<std::pair<std::pair<std::size_t, std::size_t>, float>>> results(sources.size());
        for (std::size_t cluster = 0; cluster < sources.size(); cluster++)
        {
            const auto& source = sources[cluster];
            for (std::size_t i = 0; i < source.size(); i++)
            {
                const auto point = source[i].position;
                const auto closest_point_index = find_nearest_neighbor_point_index(point, target.points);

                if (closest_point_index >= target.points.size())
                {
                    continue;
                }
                const auto closest_point = target.points[closest_point_index];
                const auto dist = glm::distance(point, closest_point);

                // if (dist < 0.1f)
                {
                    results[cluster].push_back(std::make_pair(std::make_pair(i, closest_point_index), dist));
                }
            }
        }

        return results;
    }
};

class local_geometry_correnspondance_matcher final : public correnspondance_matcher
{
public:
    virtual std::vector<std::vector<std::pair<std::pair<std::size_t, std::size_t>, float>>> find_correnspondance(
        const std::vector<std::vector<weighted_point>> &sources,
        const std::vector<std::vector<weighted_point>> &search_points,
        const point_cloud &target) override
    {
        float search_radius = 0.1f;
        std::vector<std::vector<std::pair<std::pair<std::size_t, std::size_t>, float>>> results(sources.size());

        for (std::size_t cluster = 0; cluster< sources.size(); cluster++)
        {
            const auto &source = sources[cluster];
            const auto &search_point = search_points[cluster];

            std::vector<std::vector<std::pair<point_cloud::index_type, float>>> candidates(source.size());
            for (std::size_t i = 0; i < source.size(); i++)
            {
                target.radius_search(search_point[i].position, search_radius * search_radius, candidates[i]);
            }
            for (std::size_t mi = 0; mi < source.size(); mi++)
            {
                std::size_t best_idx = 0;
                float best_cost = std::numeric_limits<float>::max();
                for (std::size_t i = 0; i < candidates[mi].size(); i++)
                {
                    const auto ui = candidates[mi][i].first;
                    float sum_cost = 0.0f;
                    std::size_t sum_count = 0;
                    for (std::size_t mj = 0; mj < source.size(); mj++)
                    {
                        if (mi == mj)
                        {
                            continue;
                        }

                        float min_cost = std::numeric_limits<float>::max();
                        float avg_cost = 0.0f;
                        int avg_count = 0;
                        for (std::size_t j = 0; j < candidates[mj].size(); j++)
                        {
                            const auto uj = candidates[mj][j].first;
                            if (ui == uj)
                            {
                                continue;
                            }

                            const auto len_diff = glm::distance(source[mi].position, source[mj].position) - glm::distance(target[ui], target[uj]);
                            const auto angle_diff = 1.0f - glm::dot(source[mi].position - source[mj].position, target[ui] - target[uj])
                                / (glm::length(source[mi].position - source[mj].position) * glm::length(target[ui] - target[uj]));
                            const auto cost = len_diff * len_diff + angle_diff * angle_diff * 1e4f;
                            min_cost = std::min(min_cost, cost);
                            avg_cost += cost;
                            avg_count++;
                        }

                        avg_cost /= avg_count;

                        if (min_cost != std::numeric_limits<float>::max())
                        {
                            sum_count++;
                            sum_cost += min_cost;
                            //sum_cost += avg_cost;
                        }
                    }

                    if (sum_count > 0)
                    {
                        sum_cost /= sum_count;
                        sum_cost += glm::distance(source[mi].position, target[ui]);
                    }

                    if (sum_cost < best_cost)
                    {
                        best_cost = sum_cost;
                        best_idx = ui;
                    }
                }

                if (best_cost < std::numeric_limits<float>::max())
                {
                    results[cluster].push_back(std::make_pair(std::make_pair(mi, best_idx), best_cost));
                }
            }
        }

        return results;
    }
};

icp_3d_3d_minimizer::icp_3d_3d_minimizer()
    : matcher(new closest_point_correnspondance_matcher())
    , elapsed(0.0)
{}

float icp_3d_3d_minimizer::update(std::size_t iter, const model_data &model, const frame_data_t &frame, const point_cloud &marker_cloud,
                                  const model_instance_data &model_instance, const std::unordered_map<std::size_t, glm::vec3> &estimated_pos, clusters_transform_params &params, bool with_twist,
                                  bool verbose)
{
    const auto &registered_clusters = model_instance.clusters;
    const bool use_quaternion = true;
    const bool use_twist = false;

    auto &mutable_rotations = params.mutable_rotations;
    auto &mutable_quat_rotations = params.mutable_quat_rotations;
    auto &mutable_translations = params.mutable_translations;
    auto &mutable_twist_angles = params.mutable_twist_angles;

    point_cloud_debug_drawer debug_point;

    std::vector<std::vector<residual_data>> residual_datas(registered_clusters.size());

    {
        const auto start = std::chrono::system_clock::now();

        const auto markers = frame.markers;
        std::vector<float> dists(markers.size());
        std::fill(dists.begin(), dists.end(), std::numeric_limits<float>::max());

        std::vector<std::vector<weighted_point>> source_points(registered_clusters.size());
        std::vector<std::vector<weighted_point>> estimated_points(registered_clusters.size());
        for (std::size_t i = 0; i < registered_clusters.size(); i++)
        {
            const auto& registered_cluster = registered_clusters[i];

            const auto twist_angle = mutable_twist_angles[i];
            // const auto rotation = &mutable_rotations[i * 3];
            const auto rotation = &mutable_quat_rotations[i * 4];
            const auto translation = &mutable_translations[i * 3];

            const auto cluster_pose = transform_to_target_pose(registered_cluster.cluster, registered_cluster.fit_result);
            const auto twist_axis = to_line(cluster_pose);
            const auto current_twist_axis = transform_quat(twist_axis, rotation, translation);

            for (std::size_t j = 0; j < registered_cluster.target.size(); j++)
            {
                const auto point = registered_cluster.target[j];

                weighted_point source_point;
                source_point.id = point.id;
                source_point.weight = point.weight;
                source_point.position = transform_twist_quat(point.position, twist_angle, twist_axis, point.weight, rotation, translation);

                source_points[i].push_back(source_point);

                auto estimated_point = source_point;
#if 0
                const auto pred_pos = estimated_pos.at(point.id);
                if (glm::distance(pred_pos, estimated_point.position) < 0.2f)
                {
                    estimated_point.position = pred_pos;
                }
#endif
                estimated_points[i].push_back(estimated_point);

#if 0
                {
                    debug_point.add(point.position, glm::u8vec3(0, 0, 255));
                    debug_point.add(pred_pos, glm::u8vec3(255, 255, 0));
                }
#endif
            }

#if 0
            {
                debug_point.add(current_twist_axis.origin, glm::u8vec3(0, 255, 255));
                debug_point.add(current_twist_axis.origin + current_twist_axis.direction * 0.3f, glm::u8vec3(0, 255, 255));
            }
#endif
        }
        const auto pairs = matcher->find_correnspondance(source_points, estimated_points, marker_cloud);

        for (std::size_t i = 0; i < registered_clusters.size(); i++)
        {
            for (const auto& p : pairs[i])
            {
                const auto pair = p.first;
                const auto point = source_points[i][pair.first].position;
                const auto target_index = pair.second;
                const auto source_index = source_points[i][pair.first].id;

                if (target_index >= markers.size())
                {
                    continue;
                }
                const auto closest_point = markers[target_index];
                const auto dist = p.second;

                dists[target_index] = std::min(dists[target_index], dist);

                residual_datas[i].push_back(residual_data{
                    point, closest_point, source_index, target_index, source_points[i][pair.first].weight, dist, 1.0f });

                if (source_index == 36)
                {
                    debug_point.add(point, glm::u8vec3(255, 255, 0));
                    debug_point.add(closest_point, glm::u8vec3(0, 255, 255));
                }
                else
                {
                    glm::u8vec3 color(255, 0, 0);
                    debug_point.add(point, color);
                }
            }
        }

        for (std::size_t i = 0; i < markers.size(); i++)
        {
            const auto marker = markers[i];
            glm::u8vec3 color(0, 255, 0);
            debug_point.add(marker, color);
        }

#if 1
        {
            std::vector<int> assignment(markers.size(), -1);
            for (std::size_t i = 0; i < residual_datas.size(); i++)
            {
                for (std::size_t j = 0; j < residual_datas[i].size(); j++)
                {
                    const auto dist = residual_datas[i][j].distance;
                    const auto point = residual_datas[i][j].source_point;
                    const auto closest_point = residual_datas[i][j].target_point;
                    const auto target_index = residual_datas[i][j].target_index;

                    if (dist == dists[target_index] && assignment[target_index] == -1)
                    {
                        assignment[target_index] = static_cast<int>(residual_datas[i][j].source_index);
                    }
                }
            }

            std::vector<std::vector<residual_data>> new_residual_datas(registered_clusters.size());
            for (std::size_t i = 0; i < residual_datas.size(); i++)
            {
                for (std::size_t j = 0; j < residual_datas[i].size(); j++)
                {
                    const auto target_index = residual_datas[i][j].target_index;
                    const auto source_index = residual_datas[i][j].source_index;

                    if (assignment[target_index] == static_cast<int>(source_index))
                    {
                        new_residual_datas[i].push_back(residual_datas[i][j]);
                    }
                }
            }

            residual_datas = new_residual_datas;
        }
#endif

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Total elapsed for labeling " << (elapsed / 1000.0) << " [ms]" << std::endl;
    }

    std::size_t num_clusters = registered_clusters.size();

#ifdef DUMP_PROBLEM
    nlohmann::json prob;
#endif

#if USE_CERES_SOLVER
    ceres::Problem problem;

    std::vector<double> rotation_params;
    std::vector<double> translation_params;
    std::vector<double> twist_angle_params;

    const size_t rotation_param_size = use_quaternion ? 4 : 3;

    for (std::size_t i = 0; i < num_clusters; i++)
    {
        if (use_quaternion)
        {
            double axis_angle[3] = { 0.0, 0.0, 0.0 };
            double q[4];
            ceres::AngleAxisToQuaternion(axis_angle, q);
            rotation_params.push_back(q[0]);
            rotation_params.push_back(q[1]);
            rotation_params.push_back(q[2]);
            rotation_params.push_back(q[3]);
        }
        else
        {
            rotation_params.push_back(0.0);
            rotation_params.push_back(0.0);
            rotation_params.push_back(0.0);
        }
        translation_params.push_back(0.0);
        translation_params.push_back(0.0);
        translation_params.push_back(0.0);
        twist_angle_params.push_back(0.0);
    }

#ifdef DUMP_PROBLEM
    {
        prob["params"]["rotation"] = rotation_params;
        prob["params"]["translation"] = translation_params;
        prob["params"]["twist_angle"] = twist_angle_params;
    }
#endif
#else
    std::map<std::string, std::shared_ptr<optimization::parameter_block>> rotate_params;
    std::map<std::string, std::shared_ptr<optimization::parameter_block>> translate_params;
    std::map<std::string, std::shared_ptr<optimization::parameter_block>> twist_angle_params;
    std::vector<std::shared_ptr<optimization::parameter_block>> parameters;
    std::vector<std::shared_ptr<optimization::function_term>> residuals;
#endif

    {
        const auto start = std::chrono::system_clock::now();

        for (std::size_t i = 0; i < num_clusters; i++)
        {
            const auto cluster_name = registered_clusters[i].cluster.name;

            const auto cluster_pose = transform_to_target_pose(registered_clusters[i].cluster, registered_clusters[i].fit_result);

            const auto rotation = &mutable_rotations[i * 3];
            const auto translation = &mutable_translations[i * 3];
            const auto twist_axis = transform_axis_angle(to_line(cluster_pose), rotation, translation);

#if USE_CERES_SOLVER
            const auto rotation_param = &rotation_params[i * rotation_param_size];
            const auto translation_param = &translation_params[i * 3];
            const auto twist_angle_param = &twist_angle_params[i];

            for (std::size_t j = 0; j < residual_datas[i].size(); j++)
            {
                const auto source_point = residual_datas[i][j].source_point;
                const auto target_point = residual_datas[i][j].target_point;
                const auto twist_weight = residual_datas[i][j].twist_weight;

#ifdef DUMP_PROBLEM
                {
                    const auto idx = prob["residual_blocks"].size();
                    prob["residual_blocks"][idx]["params"]["twist_angle"] = i;
                    prob["residual_blocks"][idx]["params"]["translation"] = i * 3;
                    prob["residual_blocks"][idx]["params"]["rotation"] = i * rotation_param_size;
                    prob["residual_blocks"][idx]["type"] = "quat_twisted_rt_transform_error";
                    prob["residual_blocks"][idx]["data"]["source_point"] = source_point;
                    prob["residual_blocks"][idx]["data"]["target_point"] = target_point;
                    prob["residual_blocks"][idx]["data"]["twist_weight"] = twist_weight;
                    prob["residual_blocks"][idx]["data"]["twist_axis"] = twist_axis;
                }
#endif

                const bool robust = true;
                ceres::LossFunction* loss = nullptr; /* squared loss */
                if (robust)
                {
                    loss = new ceres::HuberLoss(1.0);
                }

                if (!use_quaternion)
                {
                    if (use_twist)
                    {
                        ceres::CostFunction* cost_function =
                            twisted_rt_transform_error::create(source_point, target_point, twist_weight, twist_axis);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            twist_angle_param,
                            rotation_param,
                            translation_param);
                    }
                    else
                    {

                        ceres::CostFunction* cost_function =
                            rt_transform_error::create(source_point, target_point);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            rotation_param,
                            translation_param);
                    }
                }
                else
                {
                    if (use_twist)
                    {
                        ceres::CostFunction* cost_function =
                            quat_twisted_rt_transform_error::create(source_point, target_point, twist_weight, twist_axis);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            twist_angle_param,
                            rotation_param,
                            translation_param);
                    }
                    else
                    {

                        ceres::CostFunction* cost_function =
                            quat_rt_transform_error::create(source_point, target_point);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            rotation_param,
                            translation_param);
                    }
                }
            }
#else
            std::vector<glm::vec3> source_points;
            std::transform(residual_datas[i].begin(), residual_datas[i].end(), std::back_inserter(source_points), [](const auto& res)
                { return res.source_point; });

            std::vector<glm::vec3> target_points;
            std::transform(residual_datas[i].begin(), residual_datas[i].end(), std::back_inserter(target_points), [](const auto& res)
                { return res.target_point; });
            std::vector<double> twist_weights;
            std::transform(residual_datas[i].begin(), residual_datas[i].end(), std::back_inserter(twist_weights), [](const auto& res)
                { return res.twist_weight; });
            std::vector<double> weights;
            std::transform(residual_datas[i].begin(), residual_datas[i].end(), std::back_inserter(weights), [](const auto& res)
                { return res.weight_in_cluster; });

            std::shared_ptr<optimization::parameter_block> rotate_param;
            if (!use_quaternion)
            {
                rotate_param = std::make_shared<optimization::parameter_block>(0.0, 0.0, 0.0);
                rotate_params.insert(std::make_pair(cluster_name, rotate_param));
            }
            else
            {
                rotate_param = std::make_shared<optimization::parameter_block>(1.0, 0.0, 0.0, 0.0);
                rotate_params.insert(std::make_pair(cluster_name, rotate_param));
            }
            parameters.push_back(rotate_param);

            const auto translate_param = std::make_shared<optimization::parameter_block>(0.0, 0.0, 0.0);
            translate_params.insert(std::make_pair(cluster_name, translate_param));
            parameters.push_back(translate_param);

            const auto twist_angle_param = std::make_shared<optimization::parameter_block>(0.0);
            twist_angle_params.insert(std::make_pair(cluster_name, twist_angle_param));
            parameters.push_back(twist_angle_param);

            if (!use_quaternion)
            {
                if (use_twist)
                {
                    const auto functor = twisted_rt_least_square_functor(source_points, target_points, twist_weights, twist_axis, weights);
                    const auto residual = optimization::make_auto_diff_function_term(functor, rotate_param, translate_param, twist_angle_param);
                    residuals.push_back(residual);
                }
                else
                {
                    const auto functor = rt_least_square_functor(source_points, target_points, weights);
                    const auto residual = optimization::make_auto_diff_function_term(functor, rotate_param, translate_param);
                    residuals.push_back(residual);
                }
            }
            else
            {
                if (use_twist)
                {
                    const auto functor = quat_twisted_rt_least_square_functor(source_points, target_points, twist_weights, twist_axis, weights);
                    const auto residual = optimization::make_auto_diff_function_term(functor, rotate_param, translate_param, twist_angle_param);
                    residuals.push_back(residual);
                }
                else
                {
                    const auto functor = quat_rt_least_square_functor(source_points, target_points, weights);
                    const auto residual = optimization::make_auto_diff_function_term(functor, rotate_param, translate_param);
                    residuals.push_back(residual);
                }
            }
#endif
        }

        std::map<std::string, std::size_t> cluster_index_map;
        for (std::size_t j = 0; j < registered_clusters.size(); j++)
        {
            const auto& cluster = registered_clusters[j];
            cluster_index_map.insert(std::make_pair(cluster.cluster.name, j));
        }

#if 1
        {
            const auto articulation_pairs = get_articulation_pairs();
            for (std::size_t i = 0; i < articulation_pairs.size(); i++)
            {
                const auto [parent_name, child_name] = articulation_pairs[i];

                const auto parent_cluster_idx = cluster_index_map.at(parent_name);
                const auto child_cluster_idx = cluster_index_map.at(child_name);
                const auto& parent_cluster = registered_clusters[parent_cluster_idx];
                const auto& child_cluster = registered_clusters[child_cluster_idx];
                auto [pos1, rot1, pos2, rot2] = compute_articuated_pair_points(parent_cluster.cluster, child_cluster.cluster, parent_cluster.fit_result, child_cluster.fit_result);

                {
                    // const auto rotation = &mutable_rotations[parent_cluster_idx * 3];
                    const auto rotation = &mutable_quat_rotations[parent_cluster_idx * 4];
                    const auto translation = &mutable_translations[parent_cluster_idx * 3];
                    pos1 = transform_quat(pos1, rotation, translation);
                }
                {
                    // const auto rotation = &mutable_rotations[child_cluster_idx * 3];
                    const auto rotation = &mutable_quat_rotations[child_cluster_idx * 4];
                    const auto translation = &mutable_translations[child_cluster_idx * 3];
                    pos2 = transform_quat(pos2, rotation, translation);
                }

                debug_point.add(pos1, glm::u8vec3(255, 255, 0));
                debug_point.add(pos2, glm::u8vec3(0, 255, 255));

#if USE_CERES_SOLVER

                const bool robust = true;
                ceres::LossFunction *loss = nullptr; /* squared loss */
                if (robust)
                {
                    loss = new ceres::HuberLoss(1.0);
                }

                const auto parent_rotation_param = &rotation_params[parent_cluster_idx * rotation_param_size];
                const auto parent_translation_param = &translation_params[parent_cluster_idx * 3];
                const auto child_rotation_param = &rotation_params[child_cluster_idx * rotation_param_size];
                const auto child_translation_param = &translation_params[child_cluster_idx * 3];

#ifdef DUMP_PROBLEM
                {
                    const auto idx = prob["residual_blocks"].size();
                    prob["residual_blocks"][idx]["params"]["parent_rotation"] = parent_cluster_idx * rotation_param_size;
                    prob["residual_blocks"][idx]["params"]["parent_translation"] = parent_cluster_idx * 3;
                    prob["residual_blocks"][idx]["params"]["child_rotation"] = child_cluster_idx * rotation_param_size;
                    prob["residual_blocks"][idx]["params"]["child_translation"] = child_cluster_idx * 3;
                    prob["residual_blocks"][idx]["type"] = "quat_articulation_point_error";
                    prob["residual_blocks"][idx]["data"]["pos1"] = pos1;
                    prob["residual_blocks"][idx]["data"]["pos2"] = pos2;
                }
#endif

                if (use_quaternion)
                {
                    ceres::CostFunction* cost_function =
                        quat_articulation_point_error::create(pos1, pos2);

                    problem.AddResidualBlock(cost_function,
                        loss,
                        parent_rotation_param,
                        parent_translation_param,
                        child_rotation_param,
                        child_translation_param);
                }
                else
                {
                    ceres::CostFunction* cost_function =
                        articulation_point_error::create(pos1, pos2);

                    problem.AddResidualBlock(cost_function,
                        loss,
                        parent_rotation_param,
                        parent_translation_param,
                        child_rotation_param,
                        child_translation_param);
                }
#else
                const auto parent_rotate_param = rotate_params.at(parent_name);
                const auto parent_translate_param = translate_params.at(parent_name);
                const auto child_rotate_param = rotate_params.at(child_name);
                const auto child_translate_param = translate_params.at(child_name);

                if (use_quaternion)
                {
                    const auto functor = quat_rt_articulation_least_square_constraint(pos1, pos2, 1.0);
                    const auto constraint = optimization::make_auto_diff_function_term(functor,
                        parent_rotate_param, parent_translate_param, child_rotate_param, child_translate_param);
                    residuals.push_back(constraint);
                }
                else
                {
                    const auto functor = rt_articulation_least_square_constraint(pos1, pos2);
                    const auto constraint = optimization::make_auto_diff_function_term(functor,
                        parent_rotate_param, parent_translate_param, child_rotate_param, child_translate_param);
                    residuals.push_back(constraint);
                }
#endif
            }
        }
#endif

        {
            for (std::size_t i = 0; i < model.constraints.size(); i++)
            {
                const auto& point1 = model.constraints[i].point1;
                const auto& point2 = model.constraints[i].point2;

                const auto parent_name = point1.cluster_name;
                const auto child_name = point2.cluster_name;

                if (cluster_index_map.find(parent_name) == cluster_index_map.end())
                {
                    continue;
                }
                if (cluster_index_map.find(child_name) == cluster_index_map.end())
                {
                    continue;
                }

                const auto parent_cluster_idx = cluster_index_map.at(parent_name);
                const auto child_cluster_idx = cluster_index_map.at(child_name);
                const auto& parent_cluster = registered_clusters[parent_cluster_idx];
                const auto& child_cluster = registered_clusters[child_cluster_idx];
                auto pos1 = parent_cluster.target[point1.index].position;
                auto pos2 = child_cluster.target[point2.index].position;

                // 
                const auto parent_twist_angle = mutable_twist_angles[parent_cluster_idx];
                // const auto parent_rotation = &mutable_rotations[parent_cluster_idx * 3];
                const auto parent_rotation = &mutable_quat_rotations[parent_cluster_idx * 4];
                const auto parent_translation = &mutable_translations[parent_cluster_idx * 3];

                const auto parent_cluster_pose = transform_to_target_pose(parent_cluster.cluster, parent_cluster.fit_result);
                auto parent_twist_axis = to_line(parent_cluster_pose);

                const auto parent_twist_weight = parent_cluster.target[point1.index].weight;
                pos1 = transform_twist_quat(pos1, parent_twist_angle, parent_twist_axis, parent_twist_weight, parent_rotation, parent_translation);

                //
                const auto child_twist_angle = mutable_twist_angles[child_cluster_idx];
                // const auto child_rotation = &mutable_rotations[child_cluster_idx * 3];
                const auto child_rotation = &mutable_quat_rotations[child_cluster_idx * 4];
                const auto child_translation = &mutable_translations[child_cluster_idx * 3];

                const auto child_cluster_pose = transform_to_target_pose(child_cluster.cluster, child_cluster.fit_result);
                auto child_twist_axis = to_line(child_cluster_pose);

                const auto child_twist_weight = child_cluster.target[point2.index].weight;
                pos2 = transform_twist_quat(pos2, child_twist_angle, child_twist_axis, child_twist_weight, child_rotation, child_translation);

                parent_twist_axis = transform_quat(parent_twist_axis, parent_rotation, parent_translation);
                child_twist_axis = transform_quat(child_twist_axis, child_rotation, child_translation);

#if USE_CERES_SOLVER
                const auto parent_twist_angle_param = &twist_angle_params[parent_cluster_idx];
                const auto parent_rotation_param = &rotation_params[parent_cluster_idx * rotation_param_size];
                const auto parent_translation_param = &translation_params[parent_cluster_idx * 3];
                const auto child_twist_angle_param = &twist_angle_params[child_cluster_idx];
                const auto child_rotation_param = &rotation_params[child_cluster_idx * rotation_param_size];
                const auto child_translation_param = &translation_params[child_cluster_idx * 3];

#ifdef DUMP_PROBLEM
                {
                    const auto idx = prob["residual_blocks"].size();
                    prob["residual_blocks"][idx]["params"]["parent_twist_angle"] = parent_cluster_idx;
                    prob["residual_blocks"][idx]["params"]["parent_rotation"] = parent_cluster_idx * rotation_param_size;
                    prob["residual_blocks"][idx]["params"]["parent_translation"] = parent_cluster_idx * 3;
                    prob["residual_blocks"][idx]["params"]["child_twist_angle"] = child_cluster_idx;
                    prob["residual_blocks"][idx]["params"]["child_rotation"] = child_cluster_idx * rotation_param_size;
                    prob["residual_blocks"][idx]["params"]["child_translation"] = child_cluster_idx * 3;
                    prob["residual_blocks"][idx]["type"] = "quat_twist_articulation_point_error";
                    prob["residual_blocks"][idx]["data"]["pos1"] = pos1;
                    prob["residual_blocks"][idx]["data"]["pos2"] = pos2;
                    prob["residual_blocks"][idx]["data"]["parent_twist_weight"] = parent_twist_weight;
                    prob["residual_blocks"][idx]["data"]["parent_twist_axis"] = parent_twist_axis;
                    prob["residual_blocks"][idx]["data"]["child_twist_weight"] = child_twist_weight;
                    prob["residual_blocks"][idx]["data"]["child_twist_axis"] = child_twist_axis;
                }
#endif

                const bool robust = true;
                ceres::LossFunction *loss = nullptr; /* squared loss */
                if (robust)
                {
                    loss = new ceres::HuberLoss(1.0);
                }

                if (!use_quaternion)
                {
                    if (use_twist)
                    {
                        ceres::CostFunction* cost_function =
                            twist_articulation_point_error::create(pos1, pos2, parent_twist_weight, parent_twist_axis, child_twist_weight, child_twist_axis);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            parent_twist_angle_param,
                            parent_rotation_param,
                            parent_translation_param,
                            child_twist_angle_param,
                            child_rotation_param,
                            child_translation_param);
                    }
                    else
                    {
                        ceres::CostFunction* cost_function =
                            articulation_point_error::create(pos1, pos2);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            parent_rotation_param,
                            parent_translation_param,
                            child_rotation_param,
                            child_translation_param);
                    }
                }
                else
                {
                    if (use_twist)
                    {
                        ceres::CostFunction* cost_function =
                            quat_twist_articulation_point_error::create(pos1, pos2, parent_twist_weight, parent_twist_axis, child_twist_weight, child_twist_axis);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            parent_twist_angle_param,
                            parent_rotation_param,
                            parent_translation_param,
                            child_twist_angle_param,
                            child_rotation_param,
                            child_translation_param);
                    }
                    else
                    {
                        ceres::CostFunction* cost_function =
                            quat_articulation_point_error::create(pos1, pos2);

                        problem.AddResidualBlock(cost_function,
                            loss,
                            parent_rotation_param,
                            parent_translation_param,
                            child_rotation_param,
                            child_translation_param);
                    }
                }
#else
                const auto parent_rotate_param = rotate_params.at(parent_name);
                const auto parent_translate_param = translate_params.at(parent_name);
                const auto parent_twist_angle_param = twist_angle_params.at(parent_name);
                const auto child_rotate_param = rotate_params.at(child_name);
                const auto child_translate_param = translate_params.at(child_name);
                const auto child_twist_angle_param = twist_angle_params.at(child_name);

                if (!use_quaternion)
                {
                    if (use_twist)
                    {
                        const auto functor = twisted_rt_articulation_least_square_constraint(pos1, pos2, parent_twist_weight, child_twist_weight, parent_twist_axis, child_twist_axis);
                        const auto constraint = optimization::make_auto_diff_function_term(functor,
                            parent_rotate_param, parent_translate_param, parent_twist_angle_param, child_rotate_param, child_translate_param, child_twist_angle_param);
                        residuals.push_back(constraint);
                    }
                    else
                    {
                        const auto functor = rt_articulation_least_square_constraint(pos1, pos2);
                        const auto constraint = optimization::make_auto_diff_function_term(functor,
                            parent_rotate_param, parent_translate_param, child_rotate_param, child_translate_param);
                        residuals.push_back(constraint);
                    }
                }
                else
                {
                    if (use_twist)
                    {
                        const auto functor = quat_twisted_rt_articulation_least_square_constraint(pos1, pos2, parent_twist_weight, child_twist_weight, parent_twist_axis, child_twist_axis);
                        const auto constraint = optimization::make_auto_diff_function_term(functor,
                            parent_rotate_param, parent_translate_param, parent_twist_angle_param, child_rotate_param, child_translate_param, child_twist_angle_param);
                        residuals.push_back(constraint);
                    }
                    else
                    {
                        const auto functor = quat_rt_articulation_least_square_constraint(pos1, pos2);
                        const auto constraint = optimization::make_auto_diff_function_term(functor,
                            parent_rotate_param, parent_translate_param, child_rotate_param, child_translate_param);
                        residuals.push_back(constraint);
                    }
                }
#endif
            }
        }

#if USE_CERES_SOLVER
        if (use_quaternion)
        {
#if 1
            ceres::Manifold* rotation_manifold = new ceres::QuaternionManifold{};
            for (std::size_t i = 0; i < num_clusters; i++)
            {
                const auto rotate_param = &rotation_params[i * rotation_param_size];
                problem.SetManifold(rotate_param, rotation_manifold);
            }
#endif
        }

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Total elapsed for build problem " << (elapsed / 1000.0) << " [ms]" << std::endl;
    }

    // debug_point.save("temp_" + std::to_string(frame.frame_number) + "_" + std::to_string(iter) + ".pcd");

#ifdef DUMP_PROBLEM
    {
        std::ofstream ofs("./prob.json", std::ios::out);
        ofs << prob.dump(2);
    }
#endif

    float cost = 0.f;
    {
        const auto start = std::chrono::system_clock::now();

        ceres::Solver::Options options;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        // options.linear_solver_type = ceres::DENSE_QR;
        // options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.max_num_iterations = 10;

        //options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        //options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        //options.preconditioner_type = ceres::IDENTITY;
        //options.jacobi_scaling = false;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Total elapsed for minimizer : " << elapsed / 1000.0 << " [ms]" << std::endl;

        //std::cout << "Cost : " << summary.final_cost << std::endl;
        //std::cout << "Steps : " << summary.num_successful_steps << std::endl;
        //std::cout << "Residuals : " << summary.num_residual_blocks << std::endl;
        //std::cout << "Report : " << summary.FullReport() << std::endl;
        // std::cout << summary.FullReport() << std::endl;
        cost = summary.final_cost;

        {
            const auto start = std::chrono::system_clock::now();

            for (std::size_t i = 0; i < num_clusters; i++)
            {
                const auto cluster_name = registered_clusters[i].cluster.name;

                const auto rotate_param = &rotation_params[i * rotation_param_size];
                const auto translate_param = &translation_params[i * 3];
                const auto twist_angle_param = &twist_angle_params[i];

                const auto to_vec3 = [](const double* values)
                {
                    return glm::dvec3(static_cast<double>(values[0]),
                        static_cast<double>(values[1]),
                        static_cast<double>(values[2]));
                };

                const auto to_quat = [](const double* values)
                {
                    return glm::normalize(glm::dquat(static_cast<double>(values[0]),
                        static_cast<double>(values[1]),
                        static_cast<double>(values[2]),
                        static_cast<double>(values[3])));
                };

                const auto from_vec3 = [](const glm::dvec3 v, double *values)
                {
                    values[0] = v.x;
                    values[1] = v.y;
                    values[2] = v.z;
                };

                const auto from_quat = [](const glm::dquat v, double *values)
                {
                    values[0] = v.w;
                    values[1] = v.x;
                    values[2] = v.y;
                    values[3] = v.z;
                };

                const auto mutable_rotation = &mutable_rotations[i * 3];
                const auto mutable_quat_rotation = &mutable_quat_rotations[i * 4];
                const auto mutable_translation = &mutable_translations[i * 3];
                const auto mutable_twist_angle = &mutable_twist_angles[i];

                const auto angle_axis = glm::normalize(to_vec3(mutable_rotation));
                const auto len = glm::length(to_vec3(mutable_rotation));
                const auto rotation = glm::normalize(to_quat(mutable_quat_rotation));
                // const auto rotation = (std::abs(len) <= std::numeric_limits<float>::epsilon()) ? glm::dquat(1.0, 0.0, 0.0, 0.0) : glm::angleAxis(len, angle_axis);
                const auto translation = to_vec3(mutable_translation);

                glm::quat delta_rotation;

                if (!use_quaternion)
                {
                    const auto delta_angle_axis = to_vec3(rotate_param);
                    const auto delta_len = glm::length(delta_angle_axis);
                    delta_rotation = (std::abs(delta_len) <= std::numeric_limits<float>::epsilon()) ? glm::dquat(1.0, 0.0, 0.0, 0.0) : glm::angleAxis(delta_len, glm::normalize(delta_angle_axis));
                }
                else
                {
                    delta_rotation = to_quat(rotate_param);
                }
                const auto delta_translation = to_vec3(translate_param);
                const auto delta_twist_angle = twist_angle_param[0];

                srt_transform<double> current_transform(glm::dvec3(1.0), rotation, translation);
                srt_transform<double> delta_transform(glm::dvec3(1.0), delta_rotation, delta_translation);

                current_transform = delta_transform * current_transform;

                from_quat(current_transform.rotation, mutable_quat_rotation);
                from_vec3(glm::angle(current_transform.rotation) * glm::normalize(glm::axis(current_transform.rotation)), mutable_rotation);
                from_vec3(current_transform.translation, mutable_translation);

                mutable_twist_angle[0] += delta_twist_angle;
            }

            const auto end = std::chrono::system_clock::now();
            const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

            std::cout << "Total elapsed for update pose : " << elapsed / 1000.0 << " [ms]" << std::endl;
        }

        return std::abs(summary.final_cost - summary.initial_cost) / summary.initial_cost;
    }
#else
    optimization::function func(residuals);
    optimization::problem problem(func);

    for (const auto &param : parameters)
    {
        problem.add_param_block(param);
    }

    {
        const auto opt_result = problem.solve();

        std::cout << "Initial error : " << opt_result.initial_error << std::endl;
        std::cout << "Initial residual error : " << opt_result.initial_residual_error << std::endl;
        std::cout << "Final error : " << opt_result.final_error << std::endl;
        std::cout << "Final residual error : " << opt_result.final_residual_error << std::endl;
        std::cout << "Total elapsed time : " << opt_result.total_elapsed_time << std::endl;

        elapsed += opt_result.total_elapsed_time;

        for (std::size_t i = 0; i < num_clusters; i++)
        {
            const auto cluster_name = registered_clusters[i].cluster.name;

            const auto rotate_param = rotate_params.at(cluster_name);
            const auto translate_param = translate_params.at(cluster_name);
            const auto twist_angle_param = twist_angle_params.at(cluster_name);

            const auto to_vec3 = [](const double *values)
            {
                return glm::dvec3(static_cast<double>(values[0]),
                                    static_cast<double>(values[1]),
                                    static_cast<double>(values[2]));
            };

            const auto to_quat = [](const double *values)
            {
                return glm::normalize(glm::dquat(static_cast<double>(values[0]),
                                                    static_cast<double>(values[1]),
                                                    static_cast<double>(values[2]),
                                                    static_cast<double>(values[3])));
            };

            const auto from_vec3 = [](const glm::dvec3 v, double *values)
            {
                values[0] = v.x;
                values[1] = v.y;
                values[2] = v.z;
            };

            const auto mutable_rotation = &mutable_rotations[i * 3];
            const auto mutable_translation = &mutable_translations[i * 3];
            const auto mutable_twist_angle = &mutable_twist_angles[i];

            const auto angle_axis = to_vec3(mutable_rotation);
            const auto len = glm::length(angle_axis);
            const auto rotation = (std::abs(len) <= std::numeric_limits<float>::epsilon()) ? glm::dquat(1.0, 0.0, 0.0, 0.0) : glm::angleAxis(len, glm::normalize(angle_axis));
            const auto translation = to_vec3(mutable_translation);

            glm::quat delta_rotation;

            if (!use_quaternion)
            {
                const auto delta_angle_axis = to_vec3(problem.get_params(rotate_param));
                const auto delta_len = glm::length(delta_angle_axis);
                delta_rotation = (std::abs(delta_len) <= std::numeric_limits<float>::epsilon()) ? glm::dquat(1.0, 0.0, 0.0, 0.0) : glm::angleAxis(delta_len, glm::normalize(delta_angle_axis));
            }
            else
            {
                delta_rotation = to_quat(problem.get_params(rotate_param));
            }
            const auto delta_translation = to_vec3(problem.get_params(translate_param));
            const auto delta_twist_angle = problem.get_params(twist_angle_param)[0];

            srt_transform<double> current_transform(glm::dvec3(1.0), rotation, translation);
            srt_transform<double> delta_transform(glm::dvec3(1.0), delta_rotation, delta_translation);

            current_transform = delta_transform * current_transform;

            from_vec3(glm::angle(current_transform.rotation) * glm::normalize(glm::axis(current_transform.rotation)), mutable_rotation);
            from_vec3(current_transform.translation, mutable_translation);

            mutable_twist_angle[0] += delta_twist_angle;
        }
        return std::abs(opt_result.final_error - opt_result.initial_error) / opt_result.initial_error;
    }
#endif

    return 0.0;
}
