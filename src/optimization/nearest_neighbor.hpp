#pragma once

#include <vector>
#include <limits>

template <typename T>
static T find_nearest_neighbor_point(const T &point, const std::vector<T> &points, float threshold = std::numeric_limits<float>::max())
{
    if (points.size() == 0)
    {
        return T(-1);
    }
    auto min_dist = std::numeric_limits<float>::max();
    std::size_t min_idx = points.size();
    for (std::size_t i = 0; i < points.size(); i++)
    {
        const auto dist = glm::distance(point, points[i]);
        if (dist < min_dist && dist < threshold)
        {
            min_dist = dist;
            min_idx = i;
        }
    }
    if (min_idx >= points.size())
    {
        return T(-1);
    }
    return points[min_idx];
}

template <typename T>
static T find_nearest_neighbor_point(const T &point, const std::vector<T> &points, std::vector<float> &dists, float threshold = std::numeric_limits<float>::max())
{
    if (points.size() == 0)
    {
        return T(-1);
    }
    auto min_dist = std::numeric_limits<float>::max();
    std::size_t min_idx = points.size();
    for (std::size_t i = 0; i < points.size(); i++)
    {
        const auto dist = glm::distance(point, points[i]);
        if (dist < min_dist && dists[i] == std::numeric_limits<float>::max() && dist < threshold)
        {
            min_dist = dist;
            min_idx = i;
        }
    }
    if (min_idx >= points.size())
    {
        return T(-1);
    }
    dists[min_idx] = min_dist;
    return points[min_idx];
}

template <typename T>
static std::size_t find_nearest_neighbor_point_index(const T &point, const std::vector<T> &points, std::vector<float> &dists, float threshold = std::numeric_limits<float>::max())
{
    if (points.size() == 0)
    {
        return points.size();
    }
    auto min_dist = std::numeric_limits<float>::max();
    std::size_t min_idx = points.size();
    for (std::size_t i = 0; i < points.size(); i++)
    {
        const auto dist = glm::distance(point, points[i]);
        if (dist < min_dist && dist < threshold)
        {
            min_dist = dist;
            min_idx = i;
        }
    }
    if (min_idx >= points.size())
    {
        return points.size();
    }
    dists[min_idx] = std::min(dists[min_idx], min_dist);
    return min_idx;
}

template <typename T>
static std::size_t find_nearest_neighbor_point_index(const T &point, const std::vector<T> &points, float threshold = std::numeric_limits<float>::max())
{
    if (points.size() == 0)
    {
        return points.size();
    }
    auto min_dist = std::numeric_limits<float>::max();
    std::size_t min_idx = points.size();
    for (std::size_t i = 0; i < points.size(); i++)
    {
        const auto dist = glm::distance(point, points[i]);
        if (dist < min_dist && dist < threshold)
        {
            min_dist = dist;
            min_idx = i;
        }
    }
    return min_idx;
}
