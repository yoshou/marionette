#include <cstddef>
#include <vector>

#include <glm/glm.hpp>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "model.hpp"
#include "camera_info.hpp"
#include "frame.hpp"
#include "registration.hpp"
#include "transform.hpp"
#include "nearest_neighbor.hpp"
#include "model_detector.hpp"
#include "object_functions.hpp"
#include "articulation_solver.hpp"

#include "icp_3d_2d_optimizer.hpp"

#include <iostream>

struct labeled_point
{
    std::size_t cluster;
    std::size_t index;
    glm::vec2 p;
    float z;
};

static glm::vec3 project_ortho(const camera_t &camera, const glm::vec3 &pt)
{
    const auto &extrin = camera.extrin;

    const glm::mat3 proj_mat = camera.intrin.get_matrix();

    const auto view_pt = extrin.rotation * glm::vec4(pt, 1);
    const auto proj_pt = proj_mat * (glm::vec3(view_pt) / view_pt.w);

    const auto observed_pt = glm::vec2(proj_pt / proj_pt.z);
    if (observed_pt.x < 0 || observed_pt.x >= camera.width || observed_pt.y < 0 || observed_pt.y >= camera.height)
    {
        return glm::vec3(-1);
    }

    return proj_pt;
}

static void project_target(const model_instance_data &model_instance,
                           const glm::mat4 &inverse_world, const camera_t &camera, std::vector<labeled_point> &points)
{
    const auto &registered_clusters = model_instance.clusters;
    for (std::size_t i = 0; i < registered_clusters.size(); i++)
    {
        const auto &registered_cluster = registered_clusters[i];

        const auto registerd_points = registered_cluster.target;

        for (std::size_t j = 0; j < registerd_points.size(); j++)
        {
            const auto marker = transform_coordinate(registerd_points[j].position, inverse_world);
            const auto pt = project_ortho(camera, marker);

            points.push_back(labeled_point{
                i, j, glm::vec2(pt / pt.z), pt.z});
        }
    }
}

static void project_target(const model_instance_data &model_instance,
                           const glm::mat4 &inverse_world, const camera_t &camera, std::vector<labeled_point> &points, const clusters_transform_params &params)
{
    const auto &registered_clusters = model_instance.clusters;
    for (std::size_t i = 0; i < registered_clusters.size(); i++)
    {
        const auto &registered_cluster = registered_clusters[i];

        const auto &target_points = registered_cluster.target;
        const auto &cluster = registered_cluster.cluster;
        const auto &fit_result = registered_cluster.fit_result;

        for (std::size_t j = 0; j < target_points.size(); j++)
        {
            const auto marker = transform_coordinate(target_points[j].position, inverse_world);

            const auto rotation = &params.mutable_rotations[i * 3];
            const auto translation = &params.mutable_translations[i * 3];
            const auto twist_weight = target_points[j].weight;
            const auto twist_angle = params.mutable_twist_angles[i];
            const auto cluster_pose = transform_to_target_pose(cluster, fit_result);
            const auto twist_axis = to_line(inverse_world * cluster_pose);
            const auto transformed_marker = transform_twist_axis_angle(marker, twist_angle, twist_axis, twist_weight, rotation, translation);

            const auto pt = project_ortho(camera, transformed_marker);

            points.push_back(labeled_point{
                i, j, glm::vec2(pt / pt.z), pt.z});
        }
    }
}

float icp_3d_2d_minimizer::update(std::size_t iter, const model_data &model, const std::vector<camera_t> &cameras, const frame_data_t &frame,
                                  const model_instance_data &model_instance, glm::mat4 world, clusters_transform_params &params, bool with_twist, bool verbose)
{
    const auto &registered_clusters = model_instance.clusters;
    const auto inverse_world = glm::inverse(world);

    std::vector<glm::vec2> observations;
    std::vector<int> camera_index;
    std::vector<int> point_index;
    std::vector<int> cluster_index;
    std::vector<glm::vec3> points;
    std::vector<float> twist_weights;
    std::vector<float> weights;
    auto &mutable_rotations = params.mutable_rotations;
    auto &mutable_translations = params.mutable_translations;
    auto &mutable_twist_angles = params.mutable_twist_angles;

    for (std::size_t i = 0; i < registered_clusters.size(); i++)
    {
        const auto &registered_cluster = registered_clusters[i];

        const auto registerd_points = transform_coordinate(registered_cluster.target, inverse_world);

        for (std::size_t j = 0; j < registerd_points.size(); j++)
        {
            points.push_back(registerd_points[j].position);
            twist_weights.push_back(registerd_points[j].weight);
        }
    }

    std::vector<std::size_t> observed_count(points.size());

    for (std::size_t i = 0; i < cameras.size(); i++)
    {
        const auto &camera = cameras[i];
        std::vector<labeled_point> points;
        project_target(model_instance, inverse_world, camera, points, params);

#if 0
            std::vector<labeled_point> point_to_process1;
            std::vector<labeled_point> point_to_process2;
#endif
        std::vector<std::vector<float>> dists_in_cluster(registered_clusters.size());
        for (auto &dists : dists_in_cluster)
        {
            dists.resize(frame.points[i].size());
            std::fill(dists.begin(), dists.end(), std::numeric_limits<float>::max());
        }

        for (std::size_t j = 0; j < points.size(); j++)
        {
            const auto point = points[j];
            if (point.p.x < 0 || point.p.y < 0)
            {
                continue;
            }

            const auto closest_point = find_nearest_neighbor_point(point.p, frame.points[i]);
            if (closest_point.x < 0 || closest_point.y < 0)
            {
                continue;
            }
            observations.push_back(closest_point);
            camera_index.push_back(i);
            point_index.push_back(j);
            cluster_index.push_back(point.cluster);
            weights.push_back(glm::distance(closest_point, point.p));

            observed_count[j]++;

#if 0
                point_to_process1.push_back(point);
                point_to_process2.push_back(labeled_point{0, 0, closest_point});
#endif
        }

#if 0
            std::vector<std::pair<glm::vec3, glm::vec3>> lines;
            std::vector<glm::vec3> ends;
            for (std::size_t j = 0; j < registered_clusters.size(); j++)
            {
                const auto &registered_cluster = registered_clusters[j];

                const auto rotation = &mutable_rotations[j * 3];
                const auto translation = &mutable_translations[j * 3];

                const auto transform_by_param = [](glm::mat4 m, const double *axis_angle, const double *translation)
                {
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

                        return glm::translate(trans_vec) * glm::toMat4(quat) * m;
                    }
                    else
                    {
                        return glm::translate(trans_vec) * m;
                    }
                };

                const auto cluster_pose = transform_to_target_pose(registered_cluster.cluster, registered_cluster.fit_result);
                const auto twist_axis = to_line(transform_by_param(inverse_world * cluster_pose, rotation, translation));

                lines.push_back(std::make_pair(twist_axis.origin, twist_axis.origin + twist_axis.direction * 0.3f));
            }

            std::vector<std::pair<glm::vec2, glm::vec2>> proj_lines;
            for (const auto &line : lines)
            {
                const auto proj_start = project(camera, line.first);
                if (proj_start.x < 0)
                {
                    continue;
                }
                const auto proj_end = project(camera, line.second);
                if (proj_end.x < 0)
                {
                    continue;
                }
                proj_lines.push_back(std::make_pair(proj_start, proj_end));
            }

            const auto path = "point_to_process" + std::to_string(i) + "_" + std::to_string(iter) + ".png";

            if (verbose)
            {
                const auto width = camera.width;
                const auto height = camera.height;
                cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
                const auto &points1 = point_to_process1;
                const auto &points2 = point_to_process2;

                for (std::size_t i = 0; i < points1.size(); i++)
                {
                    const auto point1 = points1[i];
                    if (point1.p.x < 0 || point1.p.y < 0)
                    {
                        continue;
                    }
                    const auto point2 = points2[i];
                    if (point2.p.x < 0 || point2.p.y < 0)
                    {
                        continue;
                    }
                    cv::circle(image, cv::Point(point1.p.x, point1.p.y), 3, cv::Scalar(0, 0, 255, 255));
                    cv::circle(image, cv::Point(point2.p.x, point2.p.y), 4, cv::Scalar(0, 255, 0, 255));
                    cv::line(image, cv::Point(point1.p.x, point1.p.y), cv::Point(point2.p.x, point2.p.y), cv::Scalar(0, 255, 0, 255), 1);
                }

                for (const auto &line : proj_lines)
                {
                    const auto start = line.first;
                    const auto end = line.second;

                    cv::circle(image, cv::Point(start.x, start.y), 3, cv::Scalar(255, 0, 0, 255));
                    cv::circle(image, cv::Point(start.x, start.y), 3, cv::Scalar(255, 0, 0, 255));
                    cv::line(image, cv::Point(start.x, start.y), cv::Point(end.x, end.y), cv::Scalar(255, 0, 0, 255), 3);
                }
                cv::imwrite(path, image);
            }
            // write_point_image(path, point_to_process1, point_to_process2, camera.width, camera.height);
#endif
    }

    const auto mean_weight = std::accumulate(weights.begin(), weights.end(), 0.f) / weights.size();

    std::vector<size_t> num_cluster_cost_funcs(registered_clusters.size());
    ceres::Problem problem;
    for (std::size_t i = 0; i < observations.size(); ++i)
    {
        double *rotation = &mutable_rotations[cluster_index[i] * 3];
        double *translation = &mutable_translations[cluster_index[i] * 3];
        double *twist_angle = &mutable_twist_angles[cluster_index[i]];

        const auto view = cameras[camera_index[i]].extrin.rotation;
        const auto proj = cameras[camera_index[i]].intrin.get_matrix();
        const auto point = points[point_index[i]];
        const auto observation = observations[i];

        if (observation.x < 0)
        {
            continue;
        }

        num_cluster_cost_funcs[cluster_index[i]]++;

        const bool robust = true;
        ceres::LossFunction *loss = nullptr; /* squared loss */
        if (robust)
        {
            const auto s = std::min(1.f, mean_weight / weights[i]);
            loss = new ceres::SoftLOneLoss(s);
        }

        if (with_twist && std::abs(twist_weights[point_index[i]]) > 1e-5f)
        {
            const auto twist_weight = twist_weights[point_index[i]];
            const auto cluster_pose = transform_to_target_pose(registered_clusters[cluster_index[i]].cluster, registered_clusters[cluster_index[i]].fit_result);
            const auto twist_axis = to_line(inverse_world * cluster_pose);

            ceres::CostFunction *cost_function =
                twisted_rt_transform_projection_error::create(observation, point, view, proj, twist_weight, twist_axis);

            problem.AddResidualBlock(cost_function,
                                     loss,
                                     twist_angle,
                                     rotation,
                                     translation);
        }
        else
        {
            ceres::CostFunction *cost_function =
                rt_transform_projection_error::create(observation, point, view, proj);

            problem.AddResidualBlock(cost_function,
                                     loss,
                                     rotation,
                                     translation);
        }
    }

    const auto articulation_pairs = get_articulation_pairs();
    {

        std::map<std::string, std::size_t> cluster_index;
        for (std::size_t j = 0; j < registered_clusters.size(); j++)
        {
            const auto &cluster = registered_clusters[j];
            cluster_index.insert(std::make_pair(cluster.cluster.name, j));
        }

        for (const auto &camera : cameras)
        {
            const auto view = camera.extrin.rotation;
            const auto proj = camera.intrin.get_matrix();

            for (std::size_t i = 0; i < articulation_pairs.size(); i++)
            {
                const auto [parent_name, child_name] = articulation_pairs[i];

                const auto parent_cluster_idx = cluster_index.at(parent_name);
                const auto child_cluster_idx = cluster_index.at(child_name);
                const auto &parent_cluster = registered_clusters[parent_cluster_idx];
                const auto &child_cluster = registered_clusters[child_cluster_idx];
                const auto [pos1, rot1, pos2, rot2] = compute_articuated_pair_points(parent_cluster.cluster, child_cluster.cluster, parent_cluster.fit_result, child_cluster.fit_result);

                const auto articulation_p = transform_coordinate((pos1 + pos2) * 0.5f, inverse_world);

                ceres::CostFunction *cost_function =
                    articulation_projection_error::create(articulation_p, view, proj);

                const bool robust = false;
                ceres::LossFunction *loss = nullptr; /* squared loss */
                if (robust)
                {
                    const auto s = 1.f;
                    loss = new ceres::SoftLOneLoss(s);
                }

                const auto parent_rotation = &mutable_rotations[parent_cluster_idx * 3];
                const auto parent_translation = &mutable_translations[parent_cluster_idx * 3];
                const auto child_rotation = &mutable_rotations[child_cluster_idx * 3];
                const auto child_translation = &mutable_translations[child_cluster_idx * 3];

                problem.AddResidualBlock(cost_function,
                                         loss,
                                         parent_rotation,
                                         parent_translation,
                                         child_rotation,
                                         child_translation);
            }
        }
    }

    {
        auto start = std::chrono::system_clock::now();

        ceres::Solver::Options options;
        options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;
        // options.linear_solver_type = ceres::ITERATIVE_SCHUR;
        options.preconditioner_type = ceres::SCHUR_JACOBI;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 4;
        options.max_num_iterations = 20;
        options.function_tolerance = 1e-5;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        if (verbose)
        {
            std::cout << "Num residual blocks : " << summary.num_residual_blocks << std::endl;
            for (std::size_t i = 0; i < num_cluster_cost_funcs.size(); i++)
            {
                std::cout << "Num cost funcs of cluster " << registered_clusters[i].cluster.name << " : " << num_cluster_cost_funcs[i] << std::endl;
            }
        }
        std::cout << "Total elapsed for minimizer : " << elapsed / 1000.0 << " [ms]" << std::endl;

        std::cout << "Cost : " << summary.final_cost << std::endl;
        std::cout << "Steps : " << summary.num_successful_steps << std::endl;
        std::cout << "Residuals : " << summary.num_residual_blocks << std::endl;
        return std::abs(summary.final_cost - summary.initial_cost) / summary.initial_cost;
    }
}
