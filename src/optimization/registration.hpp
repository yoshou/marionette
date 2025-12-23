#pragma once

#include "alignment.hpp"

struct clusters_transform_params
{
    std::vector<double> mutable_rotations;
    std::vector<double> mutable_quat_rotations;
    std::vector<double> mutable_translations;
    std::vector<double> mutable_twist_angles;

    clusters_transform_params()
    {}

    static clusters_transform_params create_default(std::size_t num_clusters)
    {
        clusters_transform_params params;
        for (std::size_t i = 0; i < num_clusters; i++)
        {
            params.mutable_rotations.push_back(0.0);
            params.mutable_rotations.push_back(0.0);
            params.mutable_rotations.push_back(0.0);
            params.mutable_quat_rotations.push_back(1.0);
            params.mutable_quat_rotations.push_back(0.0);
            params.mutable_quat_rotations.push_back(0.0);
            params.mutable_quat_rotations.push_back(0.0);
            params.mutable_translations.push_back(0.0);
            params.mutable_translations.push_back(0.0);
            params.mutable_translations.push_back(0.0);
            params.mutable_twist_angles.push_back(0.0);
        }
        return params;
    }
};
