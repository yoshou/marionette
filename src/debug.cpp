#include "debug.hpp"

#include <pcl/io/pcd_io.h>

class point_cloud_debug_drawer::impl
{
public:
    pcl::PointCloud<pcl::PointXYZRGBA>::Ptr p_cloud;

    impl()
        : p_cloud(new pcl::PointCloud<pcl::PointXYZRGBA>)
    {}
};


point_cloud_debug_drawer::point_cloud_debug_drawer()
    : pimpl(new impl())
{}

point_cloud_debug_drawer::~point_cloud_debug_drawer()
{}

void point_cloud_debug_drawer::add(const glm::vec3 &point, const glm::u8vec4 &color)
{
    pcl::PointXYZRGBA p;

    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    p.r = color.r;
    p.g = color.g;
    p.b = color.b;
    p.a = color.a;
    pimpl->p_cloud->points.push_back(p);
}

void point_cloud_debug_drawer::add(const glm::vec3 &point, const glm::u8vec3 &color)
{
    pcl::PointXYZRGBA p;

    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    p.r = color.r;
    p.g = color.g;
    p.b = color.b;
    p.a = 255;
    pimpl->p_cloud->points.push_back(p);
}

void point_cloud_debug_drawer::add(const std::vector<glm::vec3> &points, const glm::u8vec4 &color)
{
    for (const auto &point : points)
    {
        add(point, color);
    }
}

void point_cloud_debug_drawer::add(const std::vector<glm::vec3> &points, const glm::u8vec3 &color)
{
    for (const auto &point : points)
    {
        add(point, color);
    }
}

void point_cloud_debug_drawer::clear()
{
    pimpl->p_cloud->points.clear();
}

std::size_t point_cloud_debug_drawer::size() const
{
    return pimpl->p_cloud->points.size();
}

void point_cloud_debug_drawer::get(std::size_t i, glm::vec3 &point, glm::u8vec3 &color) const
{
    const auto &p = pimpl->p_cloud->points[i];
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    color.r = p.r;
    color.g = p.g;
    color.b = p.b;
}

void point_cloud_debug_drawer::save(const std::string &path)
{
    pimpl->p_cloud->width = pimpl->p_cloud->points.size();
    pimpl->p_cloud->height = 1;
    pcl::io::savePCDFileASCII(path, *pimpl->p_cloud);
}

void point_cloud_debug_drawer::load(const std::string &path)
{
    pcl::io::loadPCDFile(path, *pimpl->p_cloud);
}
