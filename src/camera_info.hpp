#pragma once

#include <glm/glm.hpp>
#include <array>
#include <string>
#include <fstream>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>
#include <yaml-cpp/yaml.h>

struct camera_intrin_t
{
    float fx, fy;
    float cx, cy;
    std::array<float, 5> coeffs = {};

    glm::mat3 get_matrix() const
    {
        return glm::mat3(
            fx, 0, 0,
            0, fy, 0,
            cx, cy, 1);
    }
};

struct camera_extrin_t
{
    glm::vec3 translation;
    glm::mat4 rotation;
};

struct camera_t
{
    camera_intrin_t intrin;
    camera_extrin_t extrin;
    uint32_t width, height;
};

static void get_cv_intrinsic(const camera_intrin_t &intrin, cv::Mat &camera_matrix, cv::Mat &dist_coeffs)
{
    camera_matrix = cv::Mat::eye(3, 3, CV_64F);
    camera_matrix.at<double>(0, 0) = intrin.fx;
    camera_matrix.at<double>(1, 1) = intrin.fy;
    camera_matrix.at<double>(0, 2) = intrin.cx;
    camera_matrix.at<double>(1, 2) = intrin.cy;

    dist_coeffs = cv::Mat::zeros(5, 1, CV_64FC1);
    dist_coeffs.at<double>(0) = intrin.coeffs[0];
    dist_coeffs.at<double>(1) = intrin.coeffs[1];
    dist_coeffs.at<double>(2) = intrin.coeffs[2];
    dist_coeffs.at<double>(3) = intrin.coeffs[3];
    dist_coeffs.at<double>(4) = intrin.coeffs[4];
}

struct camera_module_t
{
    camera_t infra1;
    camera_t infra2;
    camera_t color;
};

static std::map<std::string, camera_module_t> load_camera_params(std::string path)
{
    std::ifstream ifs;
    ifs.open(path, std::ios::binary | std::ios::in);

    std::istreambuf_iterator<char> beg(ifs);
    std::istreambuf_iterator<char> end;
    std::vector<char> str_data;
    std::copy(beg, end, std::back_inserter(str_data));
    std::string str(str_data.begin(), str_data.end());

    const auto doc = YAML::Load(str.c_str());

    std::map<std::string, camera_module_t> result;

    const auto devices = doc["devices"];

    for (const auto &device : devices)
    {
        const auto serial = device["serial"].as<std::string>();

        camera_module_t param;

        auto extract = [](camera_t &dst, const YAML::Node &doc)
        {
            dst.width = doc["width"].as<float>();
            dst.height = doc["height"].as<float>();
            dst.intrin.fx = doc["fx"].as<float>();
            dst.intrin.fy = doc["fy"].as<float>();
            dst.intrin.cx = doc["ppx"].as<float>();
            dst.intrin.cy = doc["ppy"].as<float>();
            for (size_t i = 0; i < doc["coeffs"].size(); i++)
            {
                dst.intrin.coeffs[i] = doc["coeffs"][i].as<float>();
                // dst.intrin.coeffs[i] = 0;
            }
        };

        if (device["infra1"])
        {
            extract(param.infra1, device["infra1"]);
        }
        if (device["infra2"])
        {
            extract(param.infra2, device["infra2"]);
        }
        if (device["color"])
        {
            extract(param.color, device["color"]);
        }

        result[serial] = param;
    }
    return result;
}
