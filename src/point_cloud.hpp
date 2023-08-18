#pragma once
#include <glm/vec3.hpp>
#include <nanoflann.hpp>
#include <vector>

struct point_cloud
{
    using point_type = glm::vec3;
    using index_type = std::uint32_t;
    using distance_type = float;

private:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<distance_type, point_cloud>,
        point_cloud,
        3, /* dim */
        index_type
        >
        kd_tree_t;

    std::unique_ptr<kd_tree_t> index;

public:
    std::vector<point_type> points;

    inline std::size_t kdtree_get_point_count() const { return points.size(); }

    inline float kdtree_get_pt(const std::size_t idx, const std::size_t dim) const
    {
        return points[idx][dim];
    }

    template <class BoundingBox>
    inline bool kdtree_get_bbox(BoundingBox & /* bb */) const { return false; }

    point_cloud() {}
    explicit point_cloud(const std::vector<point_type> &points)
        : points(points)
    {
        index = std::make_unique<kd_tree_t>(3 /*dim*/, *this, nanoflann::KDTreeSingleIndexAdaptorParams(1024 /* max leaf */));
    }

    void build_index()
    {
        index->buildIndex();
    }

    inline point_type operator[](std::size_t index) const
    {
        return points[index];
    }

    inline std::size_t size() const
    {
        return points.size();
    }

    std::size_t knn_search(const glm::vec3 &query_pt, std::size_t num_points,
                           index_type *result_index, distance_type *result_distsq) const
    {
        return index->knnSearch(&query_pt[0], num_points, &result_index[0], &result_distsq[0]);
    }

    std::size_t radius_search(const glm::vec3 &query_pt, distance_type radius,
                              std::vector<std::pair<index_type, distance_type>> &result) const
    {
        nanoflann::SearchParameters params;
        params.sorted = true;

        std::vector<nanoflann::ResultItem<index_type, distance_type>> founds;
        const auto found_size = index->radiusSearch(&query_pt[0], radius, founds, params);

        for (const auto& found : founds)
        {
            result.push_back(std::make_pair(found.first, found.second));
        }

        return found_size;
    }
};
