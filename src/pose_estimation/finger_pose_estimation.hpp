#pragma once

#include <map>
#include <string>
#include <glm/glm.hpp>
#include "model.hpp"

void estimate_finger_pose(std::map<std::string, glm::mat4>& poses, const model_data& model);
