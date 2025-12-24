#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "motion_tracker.hpp"

namespace marionette::pose_estimation {

using namespace marionette::tracking;
using namespace marionette::optimization;

std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance);

std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance,
                                    const clusters_transform_params &params);

}  // namespace marionette::pose_estimation
