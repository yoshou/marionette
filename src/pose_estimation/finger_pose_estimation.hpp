#pragma once

#include <glm/glm.hpp>
#include <map>
#include <string>

#include "model.hpp"

namespace marionette::pose_estimation {

using namespace marionette::core;

void estimate_finger_pose(std::map<std::string, glm::mat4>& poses, const model_data& model);

}  // namespace marionette::pose_estimation
