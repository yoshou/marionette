#pragma once

#include <vector>
#include <glm/glm.hpp>

#include "model.hpp"
#include "global_registeration.hpp"

static void transform_to_target(const rigid_cluster &cluster, const find_fit_result &result, std::vector<glm::vec3> &points)
{
    const auto translation = result.translation;
    const auto rotation = result.rotation;
    const auto twist_angle = result.twist_angle;

    std::vector<glm::vec3> twisted_points;
    for (std::size_t i = 0; i < cluster.points.size(); i++)
    {
        const auto twisted_pos = twist(cluster.points[i], cluster.pose, glm::inverse(cluster.pose), twist_angle);
        twisted_points.push_back(twisted_pos);
    }

    const auto anchor_pos = cluster.points[result.anchor_index].position;
    for (const auto &twisted_pos : twisted_points)
    {
        const auto pos = rotation * (twisted_pos - anchor_pos) + translation;
        points.push_back(pos);
    }
}

static glm::mat4 compute_transform_matrix_to_target(const rigid_cluster &cluster, const find_fit_result &result)
{
    const auto translation = glm::translate(result.translation);
    const auto rotation = glm::mat4(result.rotation);

    const auto anchor_pos = cluster.points[0].position;
    const auto offset_translation = glm::translate(glm::mat4(1.f), -anchor_pos);

    const auto transform = translation * rotation * offset_translation;

    return transform;
}

static glm::mat4 transform_to_target_pose(const rigid_cluster &cluster, const find_fit_result &result)
{
    const auto source_pose = cluster.pose;
    const auto transform = compute_transform_matrix_to_target(cluster, result);
    return transform * source_pose;
}

static glm::mat4 transform_to_source_pose(const glm::mat4 &target_pose, const rigid_cluster &cluster, const find_fit_result &result)
{
    const auto transform = compute_transform_matrix_to_target(cluster, result);
    return glm::inverse(transform) * target_pose;
}

static std::vector<weighted_point> find_target_points(const rigid_cluster &cluster, const find_fit_result &result, const point_cloud &target_points)
{
    const auto translation = result.translation;
    const auto rotation = result.rotation;
    const auto twist_angle = result.twist_angle;

    const auto anchor_pos = cluster.points[0].position;

    std::vector<weighted_point> rel_source_points;
    for (const auto point : cluster.points)
    {
        rel_source_points.push_back(weighted_point{
            point.position - anchor_pos, point.weight, point.id});
    }

    const auto rel_source_pose = glm::translate(glm::mat4(1.f), -anchor_pos) * cluster.pose;

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    find_nearest_pair(rel_source_points, rel_source_pose, glm::inverse(rel_source_pose), target_points, translation, rotation, pairs, twist_angle);

    std::vector<weighted_point> paired_target_points;
    for (auto [i, j] : pairs)
    {
        paired_target_points.push_back(weighted_point{target_points[j], cluster.points[i].weight, cluster.points[i].id});
    }
    return paired_target_points;
}

static std::vector<weighted_point> find_target_points(const rigid_cluster &cluster, const find_fit_result &result, const point_cloud &target_points, std::vector<std::pair<std::size_t, std::size_t>> &pairs)
{
    const auto translation = result.translation;
    const auto rotation = result.rotation;
    const auto twist_angle = result.twist_angle;

    const auto anchor_pos = cluster.points[0].position;

    std::vector<weighted_point> rel_source_points;
    for (const auto point : cluster.points)
    {
        rel_source_points.push_back(weighted_point{
            point.position - anchor_pos, point.weight, point.id});
    }

    const auto rel_source_pose = glm::translate(glm::mat4(1.f), -anchor_pos) * cluster.pose;

    find_nearest_pair(rel_source_points, rel_source_pose, glm::inverse(rel_source_pose), target_points, translation, rotation, pairs, twist_angle);

    std::vector<weighted_point> paired_target_points;
    for (auto [i, j] : pairs)
    {
        paired_target_points.push_back(weighted_point{target_points[j], cluster.points[i].weight, cluster.points[i].id});
    }
    return paired_target_points;
}

static std::vector<weighted_point> find_source_points(const rigid_cluster &cluster, const find_fit_result &result, const point_cloud &target_points)
{
    const auto translation = result.translation;
    const auto rotation = result.rotation;
    const auto twist_angle = result.twist_angle;

    const auto anchor_pos = cluster.points[0].position;

    std::vector<weighted_point> rel_source_points;
    for (const auto point : cluster.points)
    {
        rel_source_points.push_back(weighted_point{
            point.position - anchor_pos, point.weight, point.id});
    }

    const auto rel_source_pose = glm::translate(glm::mat4(1.f), -anchor_pos) * cluster.pose;

    std::vector<std::pair<std::size_t, std::size_t>> pairs;
    find_nearest_pair(rel_source_points, rel_source_pose, glm::inverse(rel_source_pose), target_points, translation, rotation, pairs, twist_angle);

    std::vector<weighted_point> paired_target_points;
    for (auto [i, j] : pairs)
    {
        const auto target_point = target_points[j];
        const auto source_point = twist(weighted_point{glm::transpose(rotation) * (target_point - translation), cluster.points[i].weight, cluster.points[i].id}, rel_source_pose, glm::inverse(rel_source_pose), -twist_angle) + anchor_pos;
        paired_target_points.push_back(weighted_point{source_point, cluster.points[i].weight, cluster.points[i].id});
    }
    return paired_target_points;
}
