#include "retarget.hpp"

#include <nlohmann/json.hpp>
#include <fstream>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

static glm::vec3 parse_vec3(const nlohmann::json& obj)
{
    glm::vec3 result;
    for (std::size_t i = 0; i < 3; i++)
    {
        result[i] = obj[i].get<float>();
    }
    return result;
}

static glm::mat4 parse_mat4(const nlohmann::json& obj)
{
    glm::mat4 result;
    for (std::size_t i = 0; i < 4; i++)
    {
        for (std::size_t j = 0; j < 4; j++)
        {
            result[i][j] = obj[i * 4 + j].get<float>();
        }
    }
    return result;
}

void retarget_model::load(std::string path)
{
    std::ifstream ifs;
    ifs.open(path, std::ios::in | std::ios::binary);
    std::istreambuf_iterator<char> beg(ifs);
    std::istreambuf_iterator<char> end;
    std::string str(beg, end);

    const auto j = nlohmann::json::parse(str);
    const auto j_bones = j["bones"];

    for (const auto& j_bone : j_bones)
    {
        bone_data data;
        data.name = j_bone["name"].get<std::string>();
        data.pose = parse_mat4(j_bone["transform"]);
        bones.insert(std::make_pair(data.name, data));
    }
}

std::map<std::string, glm::mat4> retarget_model::compute_normalized_pose(const std::map<std::string, glm::mat4>& pose) const
{
    std::map<std::string, glm::mat4> norm_poses;

    for (auto iter = pose.begin(); iter != pose.end(); iter++)
    {
        const auto name = iter->first;
        const auto pose = iter->second;
        const auto tpose = bones.at(name).pose;
        const auto transform = glm::translate(glm::vec3(pose[3])) * glm::toMat4(glm::quat_cast(pose) * glm::inverse(glm::quat_cast(tpose)));

        norm_poses.insert(std::make_pair(name, transform));
    }

    return norm_poses;
}

glm::mat4 get_bone_global_transform(const std::string& name, const std::map<std::string, std::string>& parents, const std::map<std::string, glm::mat4>& transforms)
{
    if (transforms.find(name) == transforms.end())
    {
        return glm::mat4(1.0f);
    }
    const auto& transform = transforms.at(name);

    if (parents.find(name) == parents.end())
    {
        return transform;
    }
    return get_bone_global_transform(parents.at(name), parents, transforms) * transform;
}

glm::mat4 get_bone_local_transform(const std::string& name, const std::map<std::string, std::string>& parents, const std::map<std::string, glm::mat4>& transforms)
{
    const auto pose = transforms.at(name);
    const auto parent_pose = (parents.find(name) != parents.end() && transforms.find(parents.at(name)) != transforms.end()) ? transforms.at(parents.at(name)) : glm::mat4(1.0f);
    return glm::inverse(parent_pose) * pose;
}

std::map<std::string, glm::mat4> retarget(const std::map<std::string, glm::mat4>& pose, const std::map<std::string, glm::mat4>& dst_pose, const std::map<std::string, std::string>& parents)
{
    static const std::map<std::string, std::string> bone_name_map = {
        {"upper_arm.L", "leftUpperArm"},
        {"lower_arm.L", "leftLowerArm"},
        {"hand.L", "leftHand"},

        {"upper_arm.R", "rightUpperArm"},
        {"lower_arm.R", "rightLowerArm"},
        {"hand.R", "rightHand"},

        {"upper_leg.L", "leftUpperLeg"},
        {"lower_leg.L", "leftLowerLeg"},
        {"foot.L", "leftFoot"},

        {"upper_leg.R", "rightUpperLeg"},
        {"lower_leg.R", "rightLowerLeg"},
        {"foot.R", "rightFoot"},

        {"Chest", "chest"},
        {"Spine", "hips"},
    };

    std::map<std::string, glm::mat4> result_pose;

    for (auto iter = pose.begin(); iter != pose.end(); iter++)
    {
        const auto& name = iter->first;

        if (bone_name_map.find(name) != bone_name_map.end())
        {
            const auto target_name = bone_name_map.at(name);
            auto transform = dst_pose.at(target_name);
            {
                const auto trans = pose.at(name);

                if (parents.find(name) == parents.end()) // root bone
                {
                    for (std::size_t i = 0; i < 4; i++)
                    {
                        for (std::size_t j = 0; j < 3; j++)
                        {
                            transform[i][j] = trans[i][j];
                        }
                    }
                }
                else
                {
                    for (std::size_t i = 0; i < 3; i++)
                    {
                        for (std::size_t j = 0; j < 3; j++)
                        {
                            transform[i][j] = trans[i][j];
                        }
                    }
                }
            }

            result_pose.insert(std::make_pair(target_name, transform));
        }
    }

    return result_pose;
}
