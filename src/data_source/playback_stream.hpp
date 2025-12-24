#pragma once

#include <functional>
#include <glm/glm.hpp>
#include <regex>
#include <vector>

namespace marionette::data_source {

class playback_stream {
  std::string directory;
  std::size_t frame_no;
  std::vector<std::uint64_t> frame_numbers;
  uint32_t fps;

 public:
  playback_stream(std::string directory, uint32_t fps, std::size_t initial_frame_no = 1700);

  void subscribe_sphere(const std::string &name,
                        std::function<void(const std::vector<glm::vec3> &)> callback);
};

}  // namespace marionette::data_source
