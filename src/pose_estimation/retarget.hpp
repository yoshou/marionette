#pragma once

#include <string>
#include <map>
#include <glm/glm.hpp>

class retarget_model
{
    struct bone_data
    {
        std::string name;
        glm::mat4 pose;
    };

public:
    std::map<std::string, bone_data> bones;

    void load(std::string path);

    std::map<std::string, glm::mat4> compute_normalized_pose(const std::map<std::string, glm::mat4> &pose) const;
};

glm::mat4 get_bone_global_transform(const std::string &name, const std::map<std::string, std::string> &parents, const std::map<std::string, glm::mat4> &transforms);

glm::mat4 get_bone_local_transform(const std::string &name, const std::map<std::string, std::string> &parents, const std::map<std::string, glm::mat4> &transforms);

std::map<std::string, glm::mat4> retarget(const std::map<std::string, glm::mat4> &pose, const std::map<std::string, glm::mat4> &dst_pose, const std::map<std::string, std::string> &parents);
