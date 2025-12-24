#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <regex>
#include <vector>

namespace marionette::data_source {

class imu_playback_stream {
  std::string directory;
  std::size_t frame_no;
  std::vector<std::uint64_t> frame_numbers;
  uint32_t fps;

 public:
  imu_playback_stream(std::string directory, uint32_t fps, std::size_t initial_frame_no = 0);

  void subscribe_quat(const std::string &name,
                      std::function<void(const std::vector<glm::quat> &)> callback);
};

}  // namespace marionette::data_source
