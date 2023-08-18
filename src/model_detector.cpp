#include "model_detector.hpp"

#include <string>
#include <chrono>
#include <thread>
#include <iostream>
#include "global_registeration.hpp"
#include "point_cloud.hpp"
#include "articulation_solver.hpp"
#include "debug.hpp"

static bool solve_articulation(const model_data &model, const std::map<std::string, std::vector<find_fit_result>> &results, model_instance_data &registered_clusters, float articulation_solved_threshold = 0.05f)
{
    const auto add_articulation = [&](std::vector<articulation_results> &articulations, std::string parent_name, std::string child_name)
    {
        articulations.push_back(articulation_results(
            model.clusters.at(parent_name), model.clusters.at(child_name),
            results.at(parent_name), results.at(child_name)));
    };

    const auto add_result = [&](const std::string &cluster_name, const std::map<std::string, std::size_t> &indices)
    {
        const auto &result = results.at(cluster_name);
        registered_clusters.clusters.push_back(rigid_cluster_instance(
            model.clusters.at(cluster_name), result[indices.at(cluster_name)]));
    };

    {
        std::vector<articulation_results> articulations;
        add_articulation(articulations, "Spine", "upper_leg.R");
        add_articulation(articulations, "upper_leg.R", "lower_leg.R");
        add_articulation(articulations, "lower_leg.R", "foot.R");
        add_articulation(articulations, "Spine", "upper_leg.L");
        add_articulation(articulations, "upper_leg.L", "lower_leg.L");
        add_articulation(articulations, "lower_leg.L", "foot.L");

        auto start = std::chrono::system_clock::now();
        const auto [best_indices, error, max_error] = solve_articulation_chain_constraint(model, articulations, "foot.L", "foot.R");
        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        //std::cout << "Dijkstra " << elapsed << " [ms]" << std::endl;

        //std::cout << max_error << std::endl;

        if (max_error > articulation_solved_threshold)
        {
            return false;
        }

        add_result("Spine", best_indices);
        add_result("upper_leg.R", best_indices);
        add_result("lower_leg.R", best_indices);
        add_result("foot.R", best_indices);
        add_result("upper_leg.L", best_indices);
        add_result("lower_leg.L", best_indices);
        add_result("foot.L", best_indices);
    }
    {
        std::vector<articulation_results> articulations;
        add_articulation(articulations, "Chest", "upper_arm.R");
        add_articulation(articulations, "upper_arm.R", "lower_arm.R");
        add_articulation(articulations, "lower_arm.R", "hand.R");
        add_articulation(articulations, "Chest", "upper_arm.L");
        add_articulation(articulations, "upper_arm.L", "lower_arm.L");
        add_articulation(articulations, "lower_arm.L", "hand.L");

        const auto [best_indices, error, max_error] = solve_articulation_chain_constraint(model, articulations, "hand.L", "hand.R");

        //std::cout << max_error << std::endl;

        if (max_error > articulation_solved_threshold)
        {
            return false;
        }

        add_result("Chest", best_indices);
        add_result("upper_arm.R", best_indices);
        add_result("lower_arm.R", best_indices);
        add_result("hand.R", best_indices);
        add_result("upper_arm.L", best_indices);
        add_result("lower_arm.L", best_indices);
        add_result("hand.L", best_indices);
    }
    {
        std::vector<articulation_results> articulations;
        add_articulation(articulations, "Chest", "Neck");

        const auto [best_indices, error, max_error] = solve_articulation_chain_constraint(model, articulations, "Neck", "Chest");
        //std::cout << max_error << std::endl;

        if (max_error > articulation_solved_threshold)
        {
            return false;
        }

        add_result("Neck", best_indices);
        add_result("Chest", best_indices);
    }

    return true;
}

static const rigid_cluster_instance *get_cluster_by_name(std::string name, const model_instance_data &registered_clusters)
{
    for (const auto &cluster : registered_clusters.clusters)
    {
        if (cluster.cluster.name == name)
        {
            return &cluster;
        }
    }

    return nullptr;
}

float compute_articulation_distance(const model_instance_data &clusters)
{
    const auto articulation_pairs = get_articulation_pairs();

    std::vector<float> dists;
    for (const auto &articulation_pair : articulation_pairs)
    {
        const auto parent_cluster = get_cluster_by_name(articulation_pair.first, clusters);
        if (!parent_cluster)
        {
            continue;
        }
        const auto child_cluster = get_cluster_by_name(articulation_pair.second, clusters);
        if (!parent_cluster)
        {
            continue;
        }
        const auto [pos1, rot1, pos2, rot2] = compute_articuated_pair_points(
            parent_cluster->cluster, child_cluster->cluster, parent_cluster->fit_result, child_cluster->fit_result);
        const auto dist = glm::distance(pos1, pos2);
        dists.push_back(dist);
    }

    if (dists.size() > 0)
    {
        return *std::max_element(dists.begin(), dists.end());
    }
    return std::numeric_limits<float>::max();
}

model_instance_data detect_model(const model_data &model, const frame_data_t &frame)
{
    const std::vector<std::string> cluster_names = {
        "upper_leg.R",
        "lower_leg.R",
        "upper_leg.L",
        "lower_leg.L",
        "foot.R",
        "foot.L",
        "upper_arm.R",
        "lower_arm.R",
        "upper_arm.L",
        "lower_arm.L",
        "hand.R",
        "hand.L",
        "Chest",
        "Spine",
        "Neck",
    };

    point_cloud target_cloud(frame.markers);
    target_cloud.build_index();

    std::map<std::string, std::vector<find_fit_result>> results;
    for (const auto &cluster_name : cluster_names)
    {
        results.insert(std::make_pair(cluster_name, std::vector<find_fit_result>()));
    }

    {
        const auto start = std::chrono::system_clock::now();

#if 0
        const auto num_queries = cluster_names.size();
        std::vector<std::thread> threads;
        for (std::size_t i = 0; i < num_queries; i++)
        {
            const auto cluster_name = cluster_names[i];
            const auto &cluster = model.clusters.at(cluster_name);
            auto &result = results.at(cluster_name);

            threads.push_back(std::thread([&cluster, &target_cloud, &result]
                                          {
                                              global_registration registration;
                                              registration.find_fits(cluster, target_cloud, (cluster.max_twist_angle + cluster.min_twist_angle) * 0.5f,
                                                                     cluster.min_twist_angle, cluster.max_twist_angle, result);
                                          }));
        }

        for (auto &th : threads)
        {
            if (th.joinable())
            {
                th.join();
            }
        }
#else
        const auto num_queries = cluster_names.size();
        for (std::size_t i = 0; i < num_queries; i++)
        {
            const auto cluster_name = cluster_names[i];
            const auto& cluster = model.clusters.at(cluster_name);
            auto& result = results.at(cluster_name);

            global_registration registration;
            registration.find_fits(cluster, target_cloud, (cluster.max_twist_angle + cluster.min_twist_angle) * 0.5f,
                cluster.min_twist_angle, cluster.max_twist_angle, result);
        }
#endif

        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Total elapsed for registeration " << elapsed << " [ms]" << std::endl;
    }

    {
        auto start = std::chrono::system_clock::now();

        model_instance_data registered_clusters;
        if (!solve_articulation(model, results, registered_clusters, 0.1f))
        {
            return model_instance_data();
        }

        std::vector<std::size_t> assigned_id(frame.markers.size(), frame.markers.size());

        for (auto& cluster : registered_clusters.clusters)
        {
            std::vector<std::pair<std::size_t, std::size_t>> pairs;
            cluster.target = find_target_points(cluster.cluster, cluster.fit_result, target_cloud, pairs);

            if (pairs.size() != cluster.cluster.points.size())
            {
                return model_instance_data();
            }
            // std::cout << pairs.size() << "xxx" << cluster.cluster.points.size() << std::endl;

            for (const auto& p : pairs)
            {
                // std::cout << cluster.cluster.points[p.first].id << ", " << p.second << std::endl;
                if (assigned_id[p.second] == frame.markers.size())
                {
                    assigned_id[p.second] = cluster.cluster.points[p.first].id;
                }

                if (assigned_id[p.second] != cluster.cluster.points[p.first].id)
                {
                    return model_instance_data();
                }
            }
        }

#if 0
        point_cloud_debug_drawer debug_point;
        for (const auto& cluster_name : cluster_names)
        {
            const auto result = get_cluster_by_name(cluster_name, registered_clusters);

            std::vector<glm::vec3> registerd_points;
            transform_to_target(model.clusters.at(cluster_name), result->fit_result, registerd_points);
            debug_point.add(registerd_points, glm::u8vec3(255, 0, 0));
        }

        debug_point.add(frame.markers, glm::u8vec3(0, 255, 0));

        debug_point.save("frame" + std::to_string(point_cloud_logger::get_logger().frame) + ".pcd");
#endif

#if 0
        {
            const auto cluster_name = "foot.R";
            auto& result = results.at(cluster_name);

            std::sort(result.begin(), result.end(), [](const find_fit_result& l, const find_fit_result& r) {
                return l.error < r.error;
                });

            for (std::size_t i = 0; i < result.size(); i++)
            {
                point_cloud_debug_drawer debug_point;
                std::vector<glm::vec3> registerd_points;
                transform_to_target(model.clusters.at(cluster_name), result[i], registerd_points);
                // debug_point.add(registerd_points, glm::u8vec3(0, 0, 255));

                const auto& tmp_points = model.clusters.at(cluster_name).points;
                for (std::size_t x = 0; x < tmp_points.size(); x++)
                {
                    if (tmp_points[x].id == 38)
                    {
                        debug_point.add(registerd_points[x], glm::u8vec3(0, 255, 255));
                    }
                    else
                    {
                        debug_point.add(registerd_points[x], glm::u8vec3(0, 0, 255));
                    }
                }

                debug_point.add(frame.markers, glm::u8vec3(255, 0, 0));
                debug_point.add(glm::vec3(0.f), glm::u8vec3(0, 0, 0));

                debug_point.save("frame" + std::to_string(point_cloud_logger::get_logger().frame) + "_" + std::to_string(i) + ".pcd");
            }
        }
#endif

        auto end = std::chrono::system_clock::now();
        double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Total elapsed for articulation solver " << elapsed << " [ms]" << std::endl;

        return registered_clusters;
    }
}
