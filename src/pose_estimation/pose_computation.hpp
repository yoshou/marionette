#pragma once

#include <vector>
#include <glm/glm.hpp>
#include "motion_tracker.hpp"

std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance);

std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance, const clusters_transform_params &params);
