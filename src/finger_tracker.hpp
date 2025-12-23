#pragma once

#include <vector>
#include <string>
#include <glm/glm.hpp>
#include "model.hpp"

class finger_tracker
{
    model_data model;

public:
    std::vector<glm::mat4> sensor_to_bone;

    finger_tracker(const model_data& model);

    void initialize();

    void track(const std::vector<glm::quat>& poses);
};
