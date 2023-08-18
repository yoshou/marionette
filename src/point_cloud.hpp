#pragma once
#include <glm/vec3.hpp>
#include <nanoflann.hpp>
#include <vector>

struct point_cloud
{
    using point_type = glm::vec3;
    using index_type = std::uint32_t;

private:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, point_cloud>,
        point_cloud,
        3 /* dim */
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
                           index_type *result_index, float *result_distsq) const
    {
        return index->knnSearch(&query_pt[0], num_points, &result_index[0], &result_distsq[0]);
    }

    std::size_t radius_search(const glm::vec3 &query_pt, float radius,
                              std::vector<std::pair<index_type, float>> &result) const
    {
        nanoflann::SearchParams params;
        params.sorted = true;
        return index->radiusSearch(&query_pt[0], radius, result, params);
    }
};
