#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <glm/glm.hpp>

struct humanoid_bone
{
    std::string name;
    std::int32_t node;
    bool use_default_values;
    std::vector<double> min_values;
    std::vector<double> max_values;
    std::vector<double> center_values;
    double axis_length;
};

class model_drawer
{
public:
    model_drawer(std::string path);

    void draw(glm::mat4 wvp);

    void set_bone_transform(const std::string &name, const glm::mat4& value);
    glm::mat4 get_bone_transform(const std::string &name) const;
    glm::mat4 get_bone_global_transform(const std::string &name) const;

    std::map<std::string, glm::mat4> get_bone_transforms() const;
    void set_bone_transforms(const std::map<std::string, glm::mat4>& values);

    std::vector<std::string> get_bone_names() const
    {
        std::vector<std::string> names;
        for (const auto& [k, v] : bone_map)
        {
            names.push_back(k);
        }
        return names;
    }

    void set_blend_weight(std::string name, float value);
    float get_blend_weight(std::string name) const;

    ~model_drawer();

private:
    std::map<std::string, humanoid_bone> bone_map;
    class model_data;
    std::unique_ptr<model_data> data;
};