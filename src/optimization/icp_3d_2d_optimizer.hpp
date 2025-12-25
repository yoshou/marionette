#pragma once

#include <cstddef>
#include <vector>

#include "camera_info.hpp"
#include "frame.hpp"
#include "model.hpp"
#include "model_detector.hpp"
#include "registration.hpp"
#include "transform.hpp"

namespace marionette::optimization {

using namespace marionette::core;
using namespace marionette::tracking;

struct icp_3d_2d_minimizer {
  std::size_t frame_no;

  float update(std::size_t iter, const model_data &model, const std::vector<camera_t> &cameras,
               const frame_data_t &frame, const model_instance_data &model_instance,
               glm::mat4 world, clusters_transform_params &params, bool with_twist,
               bool verbose = true);
};

}  // namespace marionette::optimization
