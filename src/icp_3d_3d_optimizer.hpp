#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "model.hpp"
#include "camera_info.hpp"
#include "frame.hpp"
#include "registration.hpp"

class correnspondance_matcher
{
public:
    virtual std::vector<std::vector<std::pair<std::pair<std::size_t, std::size_t>, float>>> find_correnspondance(
        const std::vector<std::vector<weighted_point>> &sources,
        const std::vector<std::vector<weighted_point>> &search_points,
        const point_cloud &target) = 0;
    virtual ~correnspondance_matcher() = default;
};

struct icp_3d_3d_minimizer
{
    struct residual_data
    {
        glm::vec3 source_point;
        glm::vec3 target_point;
        std::size_t source_index;
        std::size_t target_index;
        float twist_weight;
        float distance;
        float weight_in_cluster;
    };

    std::unique_ptr<correnspondance_matcher> matcher;

    double elapsed;

    icp_3d_3d_minimizer();

    float update(std::size_t iter, const model_data &model, const frame_data_t &frame, const point_cloud &marker_cloud,
                 const model_instance_data &registered_clusters, const std::unordered_map<std::size_t, glm::vec3>& estimated_pos, clusters_transform_params &params, bool with_twist,
                 bool verbose = true);
};
