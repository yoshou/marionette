#include "motion_tracker.hpp"

#include "model_detector.hpp"
#include "registration.hpp"
#include "icp_3d_3d_optimizer.hpp"
#include "articulation_solver.hpp"
#include "debug.hpp"

#include <chrono>
#include <iostream>

static find_fit_result update_result(const find_fit_result &fit_result, const double *rotation, const double *translation, double twist_angle)
{
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

    const auto updated_pose = transform_by_param(glm::translate(fit_result.translation) * glm::mat4(fit_result.rotation), rotation, translation);

    const auto updated_twist_angle = fit_result.twist_angle + glm::degrees(static_cast<float>(twist_angle));

    return find_fit_result(
        fit_result.anchor_index,
        glm::mat3(updated_pose),
        glm::vec3(updated_pose[3]),
        updated_twist_angle,
        fit_result.error);
}

static float compute_articulation_distance(const model_instance_data &model_instance, const clusters_transform_params &params, const glm::mat4 world)
{
    const auto &clusters = model_instance.clusters;
    const auto articulation_pairs = get_articulation_pairs();

    std::map<std::string, std::size_t> cluster_index;
    for (std::size_t j = 0; j < clusters.size(); j++)
    {
        const auto &cluster = clusters[j];
        cluster_index.insert(std::make_pair(cluster.cluster.name, j));
    }

    std::vector<float> dists;
    for (const auto &articulation_pair : articulation_pairs)
    {
        const auto parent_cluster_idx = cluster_index.at(articulation_pair.first);
        const auto child_cluster_idx = cluster_index.at(articulation_pair.second);
        const auto &parent_cluster = clusters[parent_cluster_idx];
        const auto &child_cluster = clusters[child_cluster_idx];
        const auto [vec1, rot1, vec2, rot2] = compute_articuated_pair_points(parent_cluster.cluster, child_cluster.cluster, parent_cluster.fit_result, child_cluster.fit_result);

        glm::vec3 updated_vec1, updated_vec2;
        {
            const auto rotation = &params.mutable_rotations[parent_cluster_idx * 3];
            const auto translation = &params.mutable_translations[parent_cluster_idx * 3];
            const auto &cluster = parent_cluster;
            updated_vec1 = transform_axis_angle(vec1, rotation, translation);
        }
        {
            const auto rotation = &params.mutable_rotations[child_cluster_idx * 3];
            const auto translation = &params.mutable_translations[child_cluster_idx * 3];
            const auto &cluster = child_cluster;
            updated_vec2 = transform_axis_angle(vec2, rotation, translation);
        }
        const auto dist = glm::distance(updated_vec1, updated_vec2);
        dists.push_back(dist);
    }

    if (dists.size() > 0)
    {
        return *std::max_element(dists.begin(), dists.end());
    }
    return std::numeric_limits<float>::max();
}

void motion_tracker::track_key_frame(const model_data &model, const frame_data_t &frame,
                                     const frame_data_t &keyframe, const model_instance_data &keyframe_clusters, const std::unordered_map<std::size_t, glm::vec3> &estimated_pos, bool verbose, std::size_t max_iter)
{
    point_cloud marker_cloud(frame.markers);
    marker_cloud.build_index();

    const auto threshold = 0.001f;
    icp_3d_3d_minimizer minimizer;

    for (std::size_t iter = 0; iter < max_iter; iter++)
    {
        const auto cost_change = minimizer.update(iter, model, frame, marker_cloud, keyframe_clusters, estimated_pos, params, true, verbose);

        if (cost_change < threshold)
        {
            break;
        }
    }
}

void motion_tracker::track_interplated_frame(const model_data &model, const std::vector<camera_t> &cameras, const frame_data_t &frame,
                                             const frame_data_t &prev_frame, const model_instance_data &prev_clusters, glm::mat4 world, bool verbose, std::size_t max_iter)
{
    const auto threshold = 0.01f;
    // icp_3d_2d_minimizer minimizer;
    icp_3d_3d_minimizer minimizer;
    for (std::size_t iter = 0; iter < max_iter; iter++)
    {
        // const auto cost_change = minimizer.update(iter, model, cameras, frame, keyframe_clusters, world, params, true, verbose);

        // if (cost_change < threshold)
        // {
        //     break;
        // }
    }
}

model_instance_data motion_tracker::align_clusters(const model_instance_data &prev_model_instance, const point_cloud &points)
{
    const auto &prev_clusters = prev_model_instance.clusters;
    model_instance_data recovered_clusters;

    const auto &mutable_rotations = params.mutable_rotations;
    const auto &mutable_translations = params.mutable_translations;
    const auto &mutable_twist_angles = params.mutable_twist_angles;
    for (std::size_t i = 0; i < prev_clusters.size(); i++)
    {
        auto cluster = prev_clusters[i];

        const auto &prev_cluster = prev_clusters[i];
        const auto twist_angle = mutable_twist_angles[i];
        const auto rotation = &mutable_rotations[i * 3];
        const auto translation = &mutable_translations[i * 3];
        const auto cluster_pose = transform_to_target_pose(prev_cluster.cluster, prev_cluster.fit_result);
        const auto twist_axis = to_line(cluster_pose);

        std::vector<weighted_point> target_points;
        for (std::size_t j = 0; j < prev_cluster.target.size(); j++)
        {
            auto point = prev_cluster.target[j];
            const auto twist_weight = point.weight;
            const auto position = transform_twist_axis_angle(point.position, twist_angle, twist_axis, twist_weight, rotation, translation);

            // point_cloud::index_type index;
            // float dist;
            // points.knn_search(position, 1, &index, &dist);
            // if (point.weight > 0.0f && std::sqrt(dist) < 0.02f)
            // {
            //     point.position = points[index];
            // }
            // else
            {
                point.position = position;
            }

            target_points.push_back(point);
        }
        cluster.target = target_points;

        // cluster.fit_result = update_result(prev_cluster.fit_result, rotation, translation, twist_angle);
        recovered_clusters.clusters.push_back(cluster);
    }

    return recovered_clusters;
}

void motion_tracker::track_frame(const model_data &model, const frame_data_t &frame,
                                 const model_instance_data &model_instance)
{
    const auto &clusters = model_instance.clusters;
    std::cout << " Keyframe Dist : " << compute_articulation_distance(model_instance) << std::endl;

    point_cloud points(frame.markers);
    {
        const auto start = std::chrono::system_clock::now();

        points.build_index();

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Elapsed for build index " << (elapsed / 1000.0) << " [ms]" << std::endl;
    }

    std::unordered_map<std::size_t, glm::vec3> estimated_pos;
    {
        const auto start = std::chrono::system_clock::now();

        const std::size_t order = 2;
        const std::size_t n = 5;
        assert(n >= (order + 1));

        if (keyframe_clusters.clusters.size() == 0 && clusters.size() > 0)
        {
            keyframe_clusters = model_instance;
            keyframe = frame;
            params = clusters_transform_params::create_default(clusters.size());

            for (const auto &cluster : keyframe_clusters.clusters)
            {
                for (const auto &point : cluster.target)
                {
                    if (predictor.find(point.id) == predictor.end())
                    {
                        polynomial_regression pred(n, order);
                        for (std::size_t i = 0; i < n; i++)
                        {
                            pred.update(point.position);
                        }
                        predictor.insert(std::make_pair(point.id, pred));
                    }
                }
            }

#if 0
        {
            point_cloud_debug_drawer debug_points;

            std::unordered_set<std::size_t> updated;
            for (const auto &cluster : keyframe_clusters.clusters)
            {
                for (const auto &point : cluster.target)
                {
                    if (updated.find(point.id) == updated.end())
                    {
                        updated.insert(point.id);

                        auto &pred = predictor.at(point.id);
                        const auto predicted_position = pred.predict();

                        // std::cout << "diff " << point.id << " : " << glm::distance(predicted_position, point.position) << std::endl;

                        pred.update(point.position);

                        debug_points.add(point.position, glm::u8vec3(255, 0, 0));
                        debug_points.add(predicted_position, glm::u8vec3(0, 255, 0));
                    }
                }
            }

            debug_points.save("aligned_" + std::to_string(point_cloud_logger::get_logger().frame) + ".pcd");
        }
#endif
        }

        if (keyframe_clusters.clusters.size() == 0)
        {
            return;
        }

        for (const auto &cluster : keyframe_clusters.clusters)
        {
            for (const auto &point : cluster.target)
            {
                if (estimated_pos.find(point.id) == estimated_pos.end())
                {
                    auto &pred = predictor.at(point.id);
                    const auto predicted_position = pred.predict();
                    estimated_pos.insert(std::make_pair(point.id, predicted_position));
                }
            }
        }

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Elapsed for estimator " << (elapsed / 1000.0) << " [ms]" << std::endl;
    }

    {
        const auto start = std::chrono::system_clock::now();

#if 1
        track_key_frame(model, frame, keyframe, keyframe_clusters, estimated_pos, false);
#else
        track_interplated_frame(model, cameras, frame, keyframe, keyframe_clusters, world, false);

        keyframe_clusters = align_clusters(keyframe_clusters, points);
        keyframe = frame;
        params = clusters_transform_params::create_default(keyframe_clusters.size());
#endif

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Elapsed for minimize " << (elapsed / 1000.0) << " [ms]" << std::endl;
    }

    {
        const auto start = std::chrono::system_clock::now();

        const auto aligned_clusters = align_clusters(keyframe_clusters, points);

        point_cloud_debug_drawer debug_points;

        std::unordered_set<std::size_t> updated;
        for (const auto &cluster : aligned_clusters.clusters)
        {
            for (const auto &point : cluster.target)
            {
                if (updated.find(point.id) == updated.end())
                {
                    updated.insert(point.id);

                    auto &pred = predictor.at(point.id);
                    const auto predicted_position = pred.predict();

                    // std::cout << "diff " << point.id << " : " << glm::distance(predicted_position, point.position) << std::endl;

                    pred.update(point.position);

                    debug_points.add(point.position, glm::u8vec3(255, 0, 0));
                    debug_points.add(predicted_position, glm::u8vec3(0, 255, 0));
                }
            }
        }

        // debug_points.save("aligned_" + std::to_string(point_cloud_logger::get_logger().frame) + ".pcd");

        const auto end = std::chrono::system_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        std::cout << "Elapsed for alignment " << (elapsed / 1000.0) << " [ms]" << std::endl;
    }

    //std::cout << "Intraframe Dist : " << compute_articulation_distance(keyframe_clusters, params, world) << std::endl;
}
