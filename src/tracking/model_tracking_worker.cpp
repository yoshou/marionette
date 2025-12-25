#include "model_tracking_worker.hpp"

#include <iostream>

#include "point_cloud.hpp"
#include "pose_computation.hpp"
#include "registration.hpp"

namespace marionette::tracking {

using namespace marionette::core;
using namespace marionette::optimization;
using namespace marionette::pose_estimation;

model_tracking_worker::model_tracking_worker(
    const model_data& model, const model_instance_data& model_instance,
    const std::shared_ptr<marionette::core::frame_cursor>& frame_cursor)
    : running(false),
      model(model),
      instanced_model(model),
      model_instance(model_instance),
      frame_cursor(frame_cursor) {}

void model_tracking_worker::start() {
  th.reset(new std::thread(&model_tracking_worker::process, this));
}

void model_tracking_worker::stop() {
  running = false;
  if (th && th->joinable()) {
    th->join();
  }
}

std::map<std::string, glm::mat4> model_tracking_worker::get_poses() const {
  std::lock_guard lock(poses_mtx);
  return poses;
}

void model_tracking_worker::process() {
  {
    auto& clusters = model_instance.clusters;
    const auto frame = frame_cursor->get_frame();

    point_cloud target_cloud(frame.markers);
    target_cloud.build_index();
    for (auto& cluster : clusters) {
      cluster.target = find_target_points(cluster.cluster, cluster.fit_result, target_cloud);
    }

    for (std::size_t i = 0; i < clusters.size(); i++) {
      const auto sources =
          find_source_points(clusters[i].cluster, clusters[i].fit_result, target_cloud);
      instanced_model.clusters[clusters[i].cluster.name].points = sources;
    }
  }

  motion_tracker tracker;

  running = true;
  while (running) {
    frame_cursor->wait_next(running);
    frame_cursor = frame_cursor->get_next();

    std::cout << "=============== Tracking Frame : " << frame_cursor->get_frame().frame_number
              << " =================" << std::endl;

    tracker.track_frame(instanced_model, frame_cursor->get_frame(), model_instance);

    std::map<std::string, glm::mat4> poses;
    const auto intraframe_clusters = compute_pose(tracker.keyframe_clusters, tracker.params);
    for (std::size_t j = 0; j < intraframe_clusters.size(); j++) {
      const auto& fit_result = intraframe_clusters[j];

      const auto pose = fit_result;
      poses.insert(std::make_pair(tracker.keyframe_clusters.clusters[j].cluster.name, pose));
    }

    {
      std::lock_guard lock(poses_mtx);
      this->poses = poses;
    }
  }
}

}  // namespace marionette::tracking
