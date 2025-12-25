#include "pose_computation.hpp"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/transform.hpp>
#include <limits>

namespace marionette::pose_estimation {

using namespace marionette::tracking;
using namespace marionette::optimization;

std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance) {
  const auto &keyframe_clusters = model_instance.clusters;
  std::vector<glm::mat4> poses;
  for (std::size_t i = 0; i < keyframe_clusters.size(); i++) {
    const auto &cluster = keyframe_clusters[i];

    const auto pose = transform_to_target_pose(cluster.cluster, cluster.fit_result);
    poses.push_back(pose);
  }

  return poses;
}

std::vector<glm::mat4> compute_pose(const model_instance_data &model_instance,
                                    const clusters_transform_params &params) {
  const auto &keyframe_clusters = model_instance.clusters;
  std::vector<glm::mat4> poses;
  for (std::size_t i = 0; i < keyframe_clusters.size(); i++) {
    const auto &cluster = keyframe_clusters[i];

    // const auto rotation = &params.mutable_rotations[i * 3];
    const auto rotation = &params.mutable_quat_rotations[i * 4];
    const auto translation = &params.mutable_translations[i * 3];

    const auto transform_quat = [](glm::mat4 m, const double *rotation, const double *translation) {
      glm::quat quat(static_cast<float>(rotation[0]), static_cast<float>(rotation[1]),
                     static_cast<float>(rotation[2]), static_cast<float>(rotation[3]));

      glm::vec3 trans(static_cast<float>(translation[0]), static_cast<float>(translation[1]),
                      static_cast<float>(translation[2]));

      return glm::translate(trans) * glm::toMat4(quat) * m;
    };

    const auto pose = transform_to_target_pose(cluster.cluster, cluster.fit_result);
    const auto updated_pose = transform_quat(pose, rotation, translation);

    poses.push_back(updated_pose);
  }

  return poses;
}

}  // namespace marionette::pose_estimation
