#pragma once

#include <vector>
#include <utility>
#include <unordered_set>
#include <memory>

#include <glm/glm.hpp>

#include "model.hpp"
#include "transform.hpp"
#include "point_cloud.hpp"

struct twist_rt_transform
{
    glm::mat3 rotation;
    glm::vec3 translation;
    line3 twist;

    twist_rt_transform()
        : rotation(1.f), translation(0.f), twist()
    {
    }

    twist_rt_transform(const glm::mat3 &rotation, const glm::vec3 &translation, const line3 &twist)
        : rotation(rotation), translation(translation), twist(twist)
    {
    }
};

struct find_fit_result
{
    std::size_t anchor_index;
    glm::mat3 rotation;
    glm::vec3 translation;
    float twist_angle;
    float error;

    find_fit_result()
        : error(std::numeric_limits<float>::max())
    {}

    find_fit_result(std::size_t anchor_index, glm::mat3 rotation, glm::vec3 translation, float twist_angle, float error)
        : anchor_index(anchor_index), rotation(rotation), translation(translation), twist_angle(twist_angle), error(error)
    {}
};

void find_nearest_pair(
    const std::vector<weighted_point> &source_points, const glm::mat4 &source_pose, const glm::mat4 &inv_pose, const point_cloud &target_points,
    glm::vec3 translation, glm::mat3 rotation, std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle);

void find_nearest_pair(
    const std::vector<weighted_point> &source_points, const line3 &twist_axis, const point_cloud &target_points,
    const glm::vec3 &translation, const glm::mat3 &rotation, std::vector<std::pair<std::size_t, std::size_t>> &pairs, float twist_angle);

class global_registration
{
public:
    void find_fits(const rigid_cluster &cluster, const point_cloud &cloud,
                   float initial_twist_angle, float min_twist_angle, float max_twist_angle, std::vector<find_fit_result> &results);

    void find_fits(const rigid_cluster &cluster, const std::vector<glm::vec3> &points,
                   float initial_twist_angle, float min_twist_angle, float max_twist_angle, std::vector<find_fit_result> &results);
};
