#pragma once

#include <opencv2/imgproc.hpp>
#include <glm/glm.hpp>
#include <memory>

template<typename T>
static void write_point_image(std::string path, const std::vector<T> &points, std::size_t width, std::size_t height)
{
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    for (const auto point : points)
    {
        if (point.p.x < 0 || point.p.y < 0)
        {
            continue;
        }
        cv::circle(image, cv::Point(point.p.x, point.p.y), 3, cv::Scalar(255, 0, 0, 255));
    }
    cv::imwrite(path, image);
}

template<typename T>
static void write_point_image(std::string path, const std::vector<T> &points1, const std::vector<T> &points2, std::size_t width, std::size_t height)
{
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    for (std::size_t i = 0; i < points1.size(); i++)
    {
        const auto point1 = points1[i];
        if (point1.p.x < 0 || point1.p.y < 0)
        {
            continue;
        }
        cv::circle(image, cv::Point(point1.p.x, point1.p.y), 3, cv::Scalar(0, 0, 255, 255));
    }
    for (std::size_t i = 0; i < points2.size(); i++)
    {
        const auto point2 = points2[i];
        if (point2.p.x < 0 || point2.p.y < 0)
        {
            continue;
        }
        cv::circle(image, cv::Point(point2.p.x, point2.p.y), 4, cv::Scalar(0, 255, 0, 255));
    }
    cv::imwrite(path, image);
}

template<typename T>
static void write_paired_point_image(std::string path, const std::vector<T> &points1, const std::vector<T> &points2, std::size_t width, std::size_t height)
{
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    for (std::size_t i = 0; i < points1.size(); i++)
    {
        const auto point1 = points1[i];
        if (point1.p.x < 0 || point1.p.y < 0)
        {
            continue;
        }
        const auto point2 = points2[i];
        if (point2.p.x < 0 || point2.p.y < 0)
        {
            continue;
        }
        cv::circle(image, cv::Point(point1.p.x, point1.p.y), 3, cv::Scalar(0, 0, 255, 255));
        cv::circle(image, cv::Point(point2.p.x, point2.p.y), 3, cv::Scalar(255, 0, 0, 255));
        cv::line(image, cv::Point(point1.p.x, point1.p.y), cv::Point(point2.p.x, point2.p.y), cv::Scalar(0, 255, 0, 255));
    }
    cv::imwrite(path, image);
}

class point_cloud_debug_drawer final
{
    class impl;
    std::unique_ptr<impl> pimpl;

public:
    point_cloud_debug_drawer();
    ~point_cloud_debug_drawer();
    void add(const glm::vec3 &point, const glm::u8vec4 &color);
    void add(const glm::vec3 &point, const glm::u8vec3 &color);
    void add(const std::vector<glm::vec3> &points, const glm::u8vec4 &color);
    void add(const std::vector<glm::vec3> &points, const glm::u8vec3 &color);
    void clear();
    std::size_t size() const;
    void get(std::size_t i, glm::vec3 &point, glm::u8vec3 &color) const;
    void save(const std::string &path);
    void load(const std::string &path);

    std::vector<glm::mat4> poses;
};

class point_cloud_logger
{
public:
    std::size_t frame;
    
    static inline point_cloud_logger& get_logger()
    {
        static point_cloud_logger logger;
        return logger;
    }
};
