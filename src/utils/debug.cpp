#include "debug.hpp"
#include <fstream>
#include <iomanip>

namespace marionette {
namespace utils {

struct PointXYZRGBA
{
    float x, y, z;
    uint8_t r, g, b, a;
};

class point_cloud_debug_drawer::impl
{
public:
    std::vector<PointXYZRGBA> points;

    impl() {}
};


point_cloud_debug_drawer::point_cloud_debug_drawer()
    : pimpl(new impl())
{}

point_cloud_debug_drawer::~point_cloud_debug_drawer()
{}

void point_cloud_debug_drawer::add(const glm::vec3 &point, const glm::u8vec4 &color)
{
    PointXYZRGBA p;
    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    p.r = color.r;
    p.g = color.g;
    p.b = color.b;
    p.a = color.a;
    pimpl->points.push_back(p);
}

void point_cloud_debug_drawer::add(const glm::vec3 &point, const glm::u8vec3 &color)
{
    PointXYZRGBA p;
    p.x = point.x;
    p.y = point.y;
    p.z = point.z;
    p.r = color.r;
    p.g = color.g;
    p.b = color.b;
    p.a = 255;
    pimpl->points.push_back(p);
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
    pimpl->points.clear();
}

std::size_t point_cloud_debug_drawer::size() const
{
    return pimpl->points.size();
}

void point_cloud_debug_drawer::get(std::size_t i, glm::vec3 &point, glm::u8vec3 &color) const
{
    const auto &p = pimpl->points[i];
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    color.r = p.r;
    color.g = p.g;
    color.b = p.b;
}

void point_cloud_debug_drawer::save(const std::string &path)
{
    std::ofstream file(path);
    if (!file.is_open()) return;
    
    const auto &points = pimpl->points;
    file << "# .PCD v0.7 - Point Cloud Data file format\n";
    file << "VERSION 0.7\n";
    file << "FIELDS x y z rgba\n";
    file << "SIZE 4 4 4 4\n";
    file << "TYPE F F F U\n";
    file << "COUNT 1 1 1 1\n";
    file << "WIDTH " << points.size() << "\n";
    file << "HEIGHT 1\n";
    file << "VIEWPOINT 0 0 0 1 0 0 0\n";
    file << "POINTS " << points.size() << "\n";
    file << "DATA ascii\n";
    
    for (const auto &p : points)
    {
        uint32_t rgba = (static_cast<uint32_t>(p.r) << 24) | 
                        (static_cast<uint32_t>(p.g) << 16) | 
                        (static_cast<uint32_t>(p.b) << 8) | 
                        static_cast<uint32_t>(p.a);
        file << p.x << " " << p.y << " " << p.z << " " << rgba << "\n";
    }
    file.close();
}

void point_cloud_debug_drawer::load(const std::string &path)
{
    std::ifstream file(path);
    if (!file.is_open()) return;
    
    pimpl->points.clear();
    std::string line;
    bool data_section = false;
    
    while (std::getline(file, line))
    {
        if (line.find("DATA ascii") != std::string::npos)
        {
            data_section = true;
            continue;
        }
        
        if (data_section && !line.empty() && line[0] != '#')
        {
            std::istringstream iss(line);
            PointXYZRGBA p;
            uint32_t rgba;
            if (iss >> p.x >> p.y >> p.z >> rgba)
            {
                p.r = (rgba >> 24) & 0xFF;
                p.g = (rgba >> 16) & 0xFF;
                p.b = (rgba >> 8) & 0xFF;
                p.a = rgba & 0xFF;
                pimpl->points.push_back(p);
            }
        }
    }
    file.close();
}

} // namespace utils
} // namespace marionette
