#pragma once

#include <vector>
#include "registration.hpp"
#include "model.hpp"
#include "frame.hpp"

struct rigid_cluster_instance
{
    const rigid_cluster cluster;
    find_fit_result fit_result;
    std::vector<weighted_point> target;

    rigid_cluster_instance()
    {
    }

    rigid_cluster_instance(const rigid_cluster &cluster, const find_fit_result &fit_result)
        : cluster(cluster), fit_result(fit_result), target{}
    {
    }
    rigid_cluster_instance(const rigid_cluster_instance &other)
        : cluster(other.cluster), fit_result(other.fit_result), target(other.target)
    {
    }
    rigid_cluster_instance &operator=(const rigid_cluster_instance &other)
    {
        new (this) rigid_cluster_instance(other);
        return *this;
    }
};

struct model_instance_data
{
    std::vector<rigid_cluster_instance> clusters;
};

model_instance_data detect_model(const model_data &model, const frame_data_t &frame);

float compute_articulation_distance(const model_instance_data &clusters);
