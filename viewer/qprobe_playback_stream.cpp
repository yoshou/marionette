#include "qprobe_playback_stream.hpp"

#include <filesystem>
namespace fs = std::filesystem;
#include <fstream>
#include <nlohmann/json.hpp>
#include <thread>

#include "glm_json_ext.hpp"

qprobe_playback_stream::qprobe_playback_stream(std::string directory, uint32_t fps, std::size_t initial_frame_no)
    : directory(directory), frame_no(initial_frame_no), fps(fps)
{
    std::string path_pattern = "([0-9]+).json";
    for (const auto &entry : fs::directory_iterator(directory))
    {
        const std::string s = entry.path().filename().string();
        std::smatch m;
        if (std::regex_search(s, m, std::regex(path_pattern)))
        {
            if (m[1].matched)
            {
                const auto frame_no = std::stoull(m[1].str());
                frame_numbers.push_back(frame_no);
            }
        }
    }

    std::sort(frame_numbers.begin(), frame_numbers.end());
}

void qprobe_playback_stream::subscribe_quat(const std::string &name, std::function<void(const std::vector<glm::quat> &)> callback)
{
    auto next_time = std::chrono::system_clock::now() + std::chrono::duration<double>(1.0 / fps);
    while (frame_no < frame_numbers.size())
    {
        std::string path = (fs::path(directory) / (std::to_string(frame_numbers[frame_no]) + ".json")).string();

        if (!fs::exists(path))
        {
            break;
        }

        std::ifstream f;
        f.open(path, std::ios::in | std::ios::binary);
        std::string str((std::istreambuf_iterator<char>(f)),
                        std::istreambuf_iterator<char>());

        nlohmann::json j_frame = nlohmann::json::parse(str);

        std::vector<glm::quat> orientations;
        for (const auto& pose : j_frame["poses"])
        {
            orientations.push_back(pose["orientation"]);
        }

        callback(orientations);

        frame_no++;

        std::this_thread::sleep_until(next_time);
        next_time = next_time + std::chrono::duration<double>(1.0 / fps);
    }
}
