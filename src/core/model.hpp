#pragma once

#include <string>
#include <vector>
#include <map>
#include <fstream>
#include <nlohmann/json.hpp>
#include <glm/glm.hpp>

struct weighted_point
{
    glm::vec3 position;
    float weight;
    std::size_t id;

    weighted_point()
    {}
    weighted_point(const glm::vec3 &position, float weight)
        : position(position), weight(weight), id(0)
    {
    }
    weighted_point(const glm::vec3 &position, float weight, std::size_t id)
        : position(position), weight(weight), id(id)
    {
    }
};

struct rigid_cluster
{
    std::string name;
    glm::mat4 pose;
    float min_twist_angle;
    float max_twist_angle;
    std::vector<weighted_point> points;
    glm::mat4 inv_pose;

    rigid_cluster()
    {
    }

    rigid_cluster(const std::string &name, const glm::mat4 &pose, float min_twist_angle, float max_twist_angle)
        : name(name), pose(pose), min_twist_angle(min_twist_angle), max_twist_angle(max_twist_angle), points(), inv_pose(glm::inverse(pose))
    {
    }
};

class model_data
{
    struct object_data
    {
        std::string name;
        glm::vec3 position;
        std::map<std::string, float> weights;
        glm::mat4 orientation;
    };

    struct bone_data
    {
        std::string name;
        glm::mat4 pose;
        std::string anchor;
    };

    using json = nlohmann::json;

    static glm::vec3 parse_vec3(const json &obj)
    {
        glm::vec3 result;
        for (std::size_t i = 0; i < 3; i++)
        {
            result[i] = obj[i].get<float>();
        }
        return result;
    }

    static glm::mat4 parse_mat4(const json &obj)
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

public:

    std::map<std::string, rigid_cluster> clusters;
    std::vector<glm::vec3> markers;
    std::unordered_map<std::string, std::size_t> marker_ids;
    std::unordered_map<std::size_t, std::string> marker_names;

    struct cluster_point_identifier
    {
        std::string cluster_name;
        std::size_t index;
    };

    struct position_constraint
    {
        cluster_point_identifier point1;
        cluster_point_identifier point2;
    };

    std::vector<position_constraint> constraints;
    std::map<std::size_t, std::string> marker_name;

    std::vector<object_data> objects;
    std::vector<bone_data> bones;

    void load(std::string path)
    {
        std::ifstream ifs;
        ifs.open(path, std::ios::in | std::ios::binary);
        std::istreambuf_iterator<char> beg(ifs);
        std::istreambuf_iterator<char> end;
        std::string str(beg, end);

        const auto j = json::parse(str);

        const auto j_objs = j["objects"];
        const auto j_bones = j["bones"];

        for (const auto &j_obj : j_objs)
        {
            object_data data;
            data.name = j_obj["name"].get<std::string>();
            data.position = parse_vec3(j_obj["position"]);
            data.weights = j_obj["weights"].get<std::map<std::string, float>>();
            if (j_obj.contains("pose"))
            {
                data.orientation = parse_mat4(j_obj["pose"]);
            }
            objects.push_back(data);
        }

        std::map<std::string, std::string> anchor;
        for (const auto &j_bone : j_bones)
        {
            bone_data data;
            data.name = j_bone["name"].get<std::string>();
            data.pose = parse_mat4(j_bone["transform"]);
            anchor[data.name] = j_bone["anchor"].get<std::string>();
            bones.push_back(data);
        }

        for (const auto &bone : bones)
        {
            rigid_cluster cluster(bone.name, bone.pose, 0.f, 0.f);
            clusters.insert(std::make_pair(bone.name, cluster));
        }

        std::vector<std::size_t> marker_id(objects.size());
        std::iota(marker_id.begin(), marker_id.end(), 0);

        for (std::size_t i = 0; i < objects.size(); i++)
        {
            const auto &obj = objects[i];
            for (const auto &[bone_name, weight] : obj.weights)
            {
                auto &cluster = clusters[bone_name];

                if (obj.name == anchor[bone_name])
                {
                    cluster.points.insert(cluster.points.begin(), weighted_point{
                                                                      obj.position, weight, i});
                    markers.push_back(obj.position);
                }
                else
                {
                    cluster.points.push_back(weighted_point{
                        obj.position, weight, i});

                    markers.push_back(obj.position);
                }
                marker_ids[obj.name] = i;
                marker_names[i] = obj.name;
            }
        }

        for (std::size_t i = 0; i < objects.size(); i++)
        {
            const auto &obj = objects[i];
            std::vector<std::string> bone_names;
            for (const auto &[bone_name, weight] : obj.weights)
            {
                bone_names.push_back(bone_name);
            }

            if (bone_names.size() > 1)
            {
                assert(bone_names.size() == 2);
                if (bone_names[0] == "hand.R" && bone_names[1] == "lower_arm.R")
                {
                    std::swap(bone_names[0], bone_names[1]);
                }
                if (bone_names[0] == "hand.L" && bone_names[1] == "lower_arm.L")
                {
                    std::swap(bone_names[0], bone_names[1]);
                }
                auto &cluster1 = clusters[bone_names[0]];
                auto &cluster2 = clusters[bone_names[1]];

                const auto point1_iter = std::find_if(cluster1.points.begin(), cluster1.points.end(), [i](const weighted_point &p)
                                                      { return p.id == i; });
                const auto point2_iter = std::find_if(cluster2.points.begin(), cluster2.points.end(), [i](const weighted_point &p)
                                                      { return p.id == i; });

                constraints.push_back(position_constraint{
                    cluster_point_identifier{
                        bone_names[0],
                        static_cast<std::size_t>(std::distance(cluster1.points.begin(), point1_iter))
                    },
                    cluster_point_identifier{
                        bone_names[1],
                        static_cast<std::size_t>(std::distance(cluster2.points.begin(), point2_iter))
                    },
                });
            }
        }
    }
};

static std::vector<std::pair<std::string, std::string>> get_articulation_pairs()
{
    static const std::vector<std::pair<std::string, std::string>> articulation_pairs = {
        {"Spine", "upper_leg.R"},
        {"upper_leg.R", "lower_leg.R"},
        {"lower_leg.R", "foot.R"},
        {"Spine", "upper_leg.L"},
        {"upper_leg.L", "lower_leg.L"},
        {"lower_leg.L", "foot.L"},

        {"Chest", "upper_arm.R"},
        {"upper_arm.R", "lower_arm.R"},
        {"lower_arm.R", "hand.R"},
        {"Chest", "upper_arm.L"},
        {"upper_arm.L", "lower_arm.L"},
        {"lower_arm.L", "hand.L"},

        {"Chest", "Neck"},
        {"Spine", "Chest"},
    };

    return articulation_pairs;
}


class skeleton_data
{
    struct bone_data
    {
        std::string name;
        glm::mat4 pose;
    };

    using json = nlohmann::json;

    static glm::vec3 parse_vec3(const json &obj)
    {
        glm::vec3 result;
        for (std::size_t i = 0; i < 3; i++)
        {
            result[i] = obj[i].get<float>();
        }
        return result;
    }

    static glm::mat4 parse_mat4(const json &obj)
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

public:
    std::map<std::string, bone_data> bones;

    void load(std::string path)
    {
        std::ifstream ifs;
        ifs.open(path, std::ios::in | std::ios::binary);
        std::istreambuf_iterator<char> beg(ifs);
        std::istreambuf_iterator<char> end;
        std::string str(beg, end);

        const auto j = json::parse(str);

        const auto j_bones = j["bones"];

        for (const auto &j_bone : j_bones)
        {
            bone_data data;
            data.name = j_bone["name"].get<std::string>();
            data.pose = parse_mat4(j_bone["transform"]);
            bones.insert(std::make_pair(data.name, data));
        }
    }
};
