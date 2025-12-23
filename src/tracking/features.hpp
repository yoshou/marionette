#pragma once

#include <vector>
#include <array>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>

struct triangle_feature
{
    std::vector<glm::vec3> values;
};

static triangle_feature extract_triangle_feature(const glm::vec3 &keypoint, const std::vector<glm::vec3> &points, float max_radius)
{
    triangle_feature feature;

    for (std::size_t i = 0; i < points.size(); i++)
    {
        if (glm::distance2(points[i], keypoint) > (max_radius * max_radius))
        {
            continue;
        }
        for (std::size_t j = i + 1; j < points.size(); j++)
        {
            if (glm::distance2(points[j], keypoint) > (max_radius * max_radius))
            {
                continue;
            }

            const auto pt1 = keypoint;
            const auto pt2 = points[i];
            const auto pt3 = points[j];

            const auto e1 = pt2 - pt1;
            const auto e2 = pt3 - pt1;
            const auto e3 = pt3 - pt2;

            const auto len1 = glm::length(e1);
            const auto len2 = glm::length(e2);
            const auto len3 = glm::length(e3);

            feature.values.push_back(glm::vec3(std::min(len1, len2), std::max(len1, len2), len3));
        }
    }

    return feature;
}

static inline float max3(glm::vec3 v) {
  return std::max(std::max(v.x, v.y), v.z);
}

static float compute_feature_distance(const triangle_feature &feature1, const triangle_feature &feature2)
{
    if (feature1.values.size() == 0 || feature2.values.size() == 0)
    {
        return std::numeric_limits<float>::max();
    }

    std::vector<float> dists(feature1.values.size());
    for (std::size_t i = 0; i < feature1.values.size(); i++)
    {
        const auto value1 = feature1.values[i];

        float min_dist = std::numeric_limits<float>::max();
        for (const auto value2 : feature2.values)
        {
            const auto d = max3(glm::abs(value1 - value2));
            if (d < min_dist)
            {
                min_dist = d;
            }
        }

        dists[i] = min_dist;
    }
    return *std::max_element(dists.begin(), dists.end());
}

static float compute_max_distance(const std::vector<glm::vec3> &points)
{
    float max_dist = 0;
    for (std::size_t i = 0; i < points.size(); i++)
    {
        const auto anchor = points[i];
        for (std::size_t j = i + 1; j < points.size(); j++)
        {
            max_dist = std::max(max_dist, glm::distance(anchor, points[j]));
        }
    }
    return max_dist;
}

struct segment_feature
{
    std::vector<float> values;
};

static segment_feature extract_segment_feature(const glm::vec3 &keypoint, const std::vector<glm::vec3> &points, float max_radius)
{
    segment_feature feature;

    for (std::size_t i = 0; i < points.size(); i++)
    {
        const auto dist = glm::distance(points[i], keypoint);
        if (dist > max_radius)
        {
            continue;
        }
        feature.values.push_back(dist);
    }
    std::sort(feature.values.begin(), feature.values.end());

    return feature;
}

static float compute_feature_distance(const segment_feature &feature1, const segment_feature &feature2)
{
    if (feature1.values.size() == 0 || feature2.values.size() == 0)
    {
        return std::numeric_limits<float>::max();
    }

    if (feature1.values.size() <= feature2.values.size())
    {
        std::vector<float> dists(feature1.values.size());
        std::unordered_set<std::size_t> assigned;
        for (std::size_t i = 0; i < feature1.values.size(); i++)
        {
            const auto value1 = feature1.values[i];

            float min_dist = std::numeric_limits<float>::max();
            std::size_t min_idx = 0;
            for (std::size_t j = 0; j < feature2.values.size(); j++)
            {
                if (assigned.find(j) != assigned.end())
                {
                    continue;
                }
                const auto value2 = feature2.values[j];
                const auto d = std::abs(value1 - value2);
                if (d < min_dist)
                {
                    min_dist = d;
                    min_idx = j;
                }
            }

            // assigned.insert(min_idx);
            dists[i] = min_dist;
        }
        return *std::max_element(dists.begin(), dists.end());
    }
    else
    {
        std::vector<float> dists(feature2.values.size());
        std::unordered_set<std::size_t> assigned;
        for (std::size_t i = 0; i < feature2.values.size(); i++)
        {
            const auto value2 = feature2.values[i];

            float min_dist = std::numeric_limits<float>::max();
            std::size_t min_idx = 0;
            for (std::size_t j = 0; j < feature1.values.size(); j++)
            {
                const auto value1 = feature1.values[j];
                if (assigned.find(j) != assigned.end())
                {
                    continue;
                }
                const auto d = std::abs(value1 - value2);
                if (d < min_dist)
                {
                    min_dist = d;
                    min_idx = j;
                }
            }

            // assigned.insert(min_idx);
            dists[i] = min_dist;
        }
        return *std::max_element(dists.begin(), dists.end());
    }
}
