#pragma once

#include <fstream>
#include <glm/glm.hpp>
#include <glm/ext.hpp>
#include <opencv2/opencv.hpp>
#include "camera_info.hpp"

static cv::Mat glm_to_cv_mat4(const glm::mat4 &m)
{
    cv::Mat ret(4, 4, CV_32F);
    memcpy(ret.data, glm::value_ptr(m), 16 * sizeof(float));
    return ret;
}

static cv::Mat glm_to_cv_mat3x4(const glm::mat4 &m)
{
    cv::Mat ret(3, 4, CV_64F);
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            ret.at<double>(i, j) = m[j][i];
        }
    }
    return ret;
}

static cv::Mat glm_to_cv_mat3x3(const glm::mat3 &m)
{
    cv::Mat ret(3, 3, CV_64F);
    for (size_t i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < 3; j++)
        {
            ret.at<double>(i, j) = m[j][i];
        }
    }
    return ret;
}

static float distance_line_point(glm::vec3 line, glm::vec2 point)
{
    const auto a = line.x;
    const auto b = line.y;
    const auto c = line.z;

    const auto num = (a * point.x + b * point.y + c);
    const auto distsq = num * num / (a * a + b * b);

    return sqrtf(distsq);
}

static glm::vec2 project(const camera_t &camera, const glm::vec3 &pt)
{
    const auto &extrin = camera.extrin;

    const glm::mat3 proj_mat = camera.intrin.get_matrix();

    const auto k1 = camera.intrin.coeffs[0];
    const auto k2 = camera.intrin.coeffs[1];
    const auto p1 = camera.intrin.coeffs[2];
    const auto p2 = camera.intrin.coeffs[3];
    const auto k3 = camera.intrin.coeffs[4];

    const auto view_pt = extrin.rotation * glm::vec4(pt, 1);
    const auto x = view_pt.x / view_pt.z;
    const auto y = view_pt.y / view_pt.z;

    const auto r2 = x * x + y * y;
    const auto x_ = x * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                    2 * p1 * x * y + p2 * (r2 + 2 * x * x) ;
    const auto y_ = y * (1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2) +
                    2 * p2 * x * y + p1 * (r2 + 2 * y * y);
    const auto proj_pt = proj_mat * (glm::vec3(x_, y_, 1.0f) / view_pt.w);
    const auto observed_pt = glm::vec2(proj_pt / proj_pt.z);
    
    if (observed_pt.x < 0 || observed_pt.x >= camera.width || observed_pt.y < 0 || observed_pt.y >= camera.height)
    {
        return glm::vec2(-1);
    }

    return observed_pt;
}

static std::vector<std::string> split(std::string &input, char delimiter)
{
    std::istringstream stream(input);
    std::string field;
    std::vector<std::string> result;
    while (getline(stream, field, delimiter))
    {
        result.push_back(field);
    }
    return result;
}

static void read_csv(std::string filename, std::vector<std::vector<std::string>> &result)
{
    std::ifstream ifs;
    ifs.open(filename, std::ios::in);
    if (!ifs)
    {
        std::cout << "File does not exist" << std::endl;
        return;
    }

    int cols = -1;

    std::string str;
    while (getline(ifs, str))
    {
        std::size_t col = 0;
        const auto strs = split(str, ',');
        if (cols == -1)
        {
            cols = static_cast<int>(strs.size());
            result.resize(cols);
            for (std::size_t i = 0; i < result.size(); i++)
            {
                result[i].clear();
            }
        }
        if (strs.size() != static_cast<std::size_t>(cols))
        {
            std::cout << "Invalid col size" << std::endl;
            return;
        }

        for (auto s : strs)
        {
            result[col++].push_back(s);
        }
    }
}
