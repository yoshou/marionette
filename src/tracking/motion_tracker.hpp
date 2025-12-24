#pragma once

#include <glm/glm.hpp>
#include <vector>

#include "camera_info.hpp"
#include "frame.hpp"
#include "model.hpp"
#include "model_detector.hpp"
#include "motion_estimation.hpp"
#include "registration.hpp"

namespace marionette::tracking {

using namespace marionette::core;

class motion_tracker {
 public:
  model_instance_data keyframe_clusters;
  clusters_transform_params params;
  frame_data_t keyframe;
  std::map<std::size_t, polynomial_regression> predictor;

  void track_key_frame(const model_data &model, const frame_data_t &frame,
                       const frame_data_t &keyframe, const model_instance_data &keyframe_clusters,
                       const std::unordered_map<std::size_t, glm::vec3> &estimated_pos,
                       bool verbose, std::size_t max_iter = 5);

  void track_interplated_frame(const model_data &model, const std::vector<camera_t> &cameras,
                               const frame_data_t &frame, const frame_data_t &prev_frame,
                               const model_instance_data &prev_clusters, glm::mat4 world,
                               bool verbose, std::size_t max_iter = 5);

  model_instance_data align_clusters(const model_instance_data &prev_model_instance,
                                     const point_cloud &points);

  void track_frame(const model_data &model, const frame_data_t &frame,
                   const model_instance_data &model_instance);
};

}  // namespace marionette::tracking
