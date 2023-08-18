#pragma once

#include <vector>
#include <fstream>
#include <string>
#include <iterator>
#include <glm/glm.hpp>
#include <nlohmann/json.hpp>
#include "glm_json_ext.hpp"

struct frame_data_t
{
    std::vector<glm::vec3> markers;
    std::vector<std::vector<glm::vec2>> points;
    uint64_t frame_number;
};

static void load_frames(std::string path, std::vector<frame_data_t> &frames, const glm::mat4 &transform)
{
    std::ifstream ifs;
    ifs.open(path, std::ios::binary | std::ios::in);
    using json = nlohmann::json;

    std::istreambuf_iterator<char> beg(ifs);
    std::istreambuf_iterator<char> end;
    std::string str(beg, end);

    const auto j = json::parse(str);

    for (size_t i = 0; i < j.size(); i++)
    {
        const auto& j_frame = j[i];
        std::vector<glm::vec3> markers;
        for (const auto &j_point : j_frame["markers"])
        {
            const auto point = j_point.get<glm::vec3>();
            markers.push_back(glm::vec3(transform * glm::vec4(point, 1.f)));
        }

        const auto points = j_frame["points"].get<std::vector<std::vector<glm::vec2>>>();
        frames.push_back(frame_data_t{markers, points, i});
    }
}
