#pragma once

#include <map>
#include <string>
#include <glm/glm.hpp>
#include "model.hpp"

using namespace marionette::core;

void estimate_finger_pose(std::map<std::string, glm::mat4>& poses, const model_data& model);
