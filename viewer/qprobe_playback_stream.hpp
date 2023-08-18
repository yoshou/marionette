#pragma once

#include <regex>
#include <functional>
#include <vector>
#include <glm/glm.hpp>

class qprobe_playback_stream
{
    std::string directory;
    std::size_t frame_no;
    std::vector<std::uint64_t> frame_numbers;
    uint32_t fps;

public:
    qprobe_playback_stream(std::string directory, uint32_t fps, std::size_t initial_frame_no = 0);

    void subscribe_quat(const std::string &name, std::function<void(const std::vector<glm::quat> &)> callback);
};
