#include "finger_tracker.hpp"

#include <algorithm>
#include <stdexcept>

namespace marionette::tracking {

finger_tracker::finger_tracker(const model_data& model) : model(model) { initialize(); }

void finger_tracker::initialize() {
  std::vector<std::string> sensors = {"Cube.004", "Cube.003", "Cube.002",
                                      "Cube.001", "Cube.005", "Cube"};
  std::vector<std::string> bones = {"Little Distal.R", "Ring Distal.R",  "Middle Distal.R",
                                    "Index Distal.R",  "Thumb Distal.R", "hand.R"};

  for (size_t i = 0; i < sensors.size(); i++) {
    const auto sensor = std::find_if(model.objects.begin(), model.objects.end(),
                                     [&](const auto& obj) { return obj.name == sensors[i]; });
    if (sensor == model.objects.end()) {
      throw std::runtime_error("Invalid model");
    }
    const auto bone = std::find_if(model.bones.begin(), model.bones.end(),
                                   [&](const auto& obj) { return obj.name == bones[i]; });
    if (bone == model.bones.end()) {
      throw std::runtime_error("Invalid model");
    }
    glm::mat4 sensor_pose = sensor->orientation;
    for (size_t k = 0; k < 3; k++) {
      sensor_pose[k] = glm::normalize(sensor_pose[k]);
    }
    sensor_pose[3] = glm::vec4(sensor->position, 1.f);

    glm::mat4 bone_pose = bone->pose;
    for (size_t k = 0; k < 3; k++) {
      bone_pose[k] = glm::normalize(bone_pose[k]);
    }

    sensor_to_bone.push_back(glm::inverse(sensor_pose) * bone_pose);
  }
}

void finger_tracker::track(const std::vector<glm::quat>& poses) {
  // Implementation placeholder
}

}  // namespace marionette::tracking
