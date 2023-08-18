
#define _USE_MATH_DEFINES

#include <memory>
#include <iostream>
#include <vector>
#include <fstream>
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <nlohmann/json.hpp>
#include <boost/program_options.hpp>

#include "fbx_loader.hpp"

static nlohmann::json mat4_to_json(const glm::mat4 &m)
{
    std::array<float, 16> values;
    const auto ptr = static_cast<const float *>(glm::value_ptr(m));
    std::copy_n(ptr, 16, values.begin());
    nlohmann::json j = values;
    return j;
}

static nlohmann::json vec3_to_json(const glm::vec3 &v)
{
    std::array<float, 3> values;
    const auto ptr = static_cast<const float *>(glm::value_ptr(v));
    std::copy_n(ptr, 3, values.begin());
    nlohmann::json j = values;
    return j;
};

int convert_tpose(int argc, const char *argv[])
{
    using namespace marionette::model;

    const auto output_name = "../data/DefaultPose.json";
    const auto input_name = "../data/FullTrackingBone.fbx";

    const auto processed_node = marionette::model::fbx::load_model(input_name);

    std::vector<std::shared_ptr<skeleton_node_t>> skeletons;
    traverse_node(processed_node, [&](const std::shared_ptr<node_t> &node)
                  {
        if (const auto skeleton = std::dynamic_pointer_cast<skeleton_node_t>(node))
        {
            skeletons.push_back(skeleton);
        } });

    using json = nlohmann::json;

    constexpr auto scale = 0.01f;

    const auto world_node = std::make_shared<node_t>();
    world_node->children.push_back(processed_node);
    processed_node->parent = world_node;
    world_node->transform = glm::scale(glm::vec3(scale));

    std::vector<json> j_bones;
    for (const auto &skeleton : skeletons)
    {
        json j_bone;
        j_bone["name"] = skeleton->name;
        j_bone["transform"] = mat4_to_json(skeleton->parent.lock()->calculate_absolute_transform());
        j_bones.push_back(j_bone);
    }

    json j = {
        {"bones", j_bones},
    };

    //std::ofstream ofs;
    //ofs.open(output_name, std::ios::out | std::ios::binary);
    //ofs << j.dump(2);

    std::cout << "Successfully converted." << std::endl;

    return 0;
}

int convert_model(int argc, const char *argv[])
{
    using namespace marionette::model;

    const auto input_name = "../data/TrackingModel.fbx";
    const auto output_name = "../data/TrackingModel.json";

    const auto processed_node = marionette::model::fbx::load_model(input_name);

    std::vector<std::shared_ptr<skeleton_node_t>> skeletons;
    std::vector<std::shared_ptr<mesh_node_t>> meshs;

    traverse_node(processed_node, [&](const std::shared_ptr<node_t> &node)
                  {
        if (const auto skeleton = std::dynamic_pointer_cast<skeleton_node_t>(node))
        {
            skeletons.push_back(skeleton);
        }
        else if (const auto mesh = std::dynamic_pointer_cast<mesh_node_t>(node))
        {
            meshs.push_back(mesh);
        } });

    using json = nlohmann::json;

    constexpr auto scale = 0.01f;

    const auto world_node = std::make_shared<node_t>();
    world_node->children.push_back(processed_node);
    processed_node->parent = world_node;
    world_node->transform = glm::scale(glm::vec3(scale));

    std::map<std::string, std::map<std::string, float>> weights = {
        {"Marker_R0", {
                          {"Chest", 0.0f},
                          {"upper_arm.R", 0.0f},
                      }},
        {"Marker_R1", {
                          {"Chest", 0.0f},
                          {"upper_arm.R", 0.0f},
                      }},
        {"Marker_R2", {
                          {"upper_arm.R", 0.8f},
                      }},
        {"Marker_R3", {
                          {"upper_arm.R", 1.0f},
                          {"lower_arm.R", 0.0f},
                      }},
        {"Marker_R4", {
                          {"upper_arm.R", 1.0f},
                          {"lower_arm.R", 0.0f},
                      }},
        {"Marker_R5", {
                          {"lower_arm.R", 0.5f},
                      }},
        {"Marker_R6", {
                          {"lower_arm.R", 1.0f},
                          {"hand.R", 0.0f},
                      }},
        {"Marker_R7", {
                          {"lower_arm.R", 1.0f},
                          {"hand.R", 0.0f},
                      }},
        {"Marker_R8", {
                          {"Spine", 0.0f},
                      }},
        {"Marker_R9", {
                          {"Spine", 0.0f},
                          {"upper_leg.R", 0.0f},
                      }},
        {"Marker_R10", {
                           {"upper_leg.R", 0.8f},
                       }},
        {"Marker_R11", {
                           {"upper_leg.R", 1.0f},
                           {"lower_leg.R", 0.0f},
                       }},
        {"Marker_R12", {
                           {"upper_leg.R", 1.0f},
                           {"lower_leg.R", 0.0f},
                       }},
        {"Marker_R13", {
                           {"lower_leg.R", 1.0f},
                       }},
        {"Marker_R14", {
                           {"lower_leg.R", 1.0f},
                           {"foot.R", 0.0f},
                       }},
        {"Marker_R15", {
                           {"foot.R", 0.0f},
                       }},
        {"Marker_R16", {
                           {"foot.R", 0.0f},
                       }},
        {"Marker_R17", {
                           {"foot.R", 0.0f},
                       }},
        {"Marker_R18", {
                           {"lower_leg.R", 1.0f},
                           {"foot.R", 0.0f},
                       }},
        {"Marker_R19", {
                           {"hand.R", 0.0f},
                       }},
        {"Marker_R20", {
                           {"hand.R", 0.0f},
                       }},

        {"Marker_L0", {
                          {"Chest", 0.0f},
                          {"upper_arm.L", 0.0f},
                      }},
        {"Marker_L1", {
                          {"Chest", 0.0f},
                          {"upper_arm.L", 0.0f},
                      }},
        {"Marker_L2", {
                          {"upper_arm.L", 0.3f},
                      }},
        {"Marker_L3", {
                          {"upper_arm.L", 1.0f},
                          {"lower_arm.L", 0.0f},
                      }},
        {"Marker_L4", {
                          {"upper_arm.L", 1.0f},
                          {"lower_arm.L", 0.0f},
                      }},
        {"Marker_L5", {
                          {"lower_arm.L", 0.5f},
                      }},
        {"Marker_L6", {
                          {"lower_arm.L", 1.0f},
                          {"hand.L", 0.0f},
                      }},
        {"Marker_L7", {
                          {"lower_arm.L", 1.0f},
                          {"hand.L", 0.0f},
                      }},
        {"Marker_L8", {
                          {"Spine", 0.0f},
                      }},
        {"Marker_L9", {
                          {"upper_leg.L", 0.0f},
                          {"Spine", 0.0f},
                      }},
        {"Marker_L10", {
                           {"upper_leg.L", 0.5f},
                       }},
        {"Marker_L11", {
                           {"upper_leg.L", 1.0f},
                           {"lower_leg.L", 0.0f},
                       }},
        {"Marker_L12", {
                           {"upper_leg.L", 1.0f},
                           {"lower_leg.L", 0.0f},
                       }},
        {"Marker_L13", {
                           {"lower_leg.L", 1.0f},
                       }},
        {"Marker_L14", {
                           {"lower_leg.L", 1.0f},
                           {"foot.L", 0.0f},
                       }},
        {"Marker_L15", {
                           {"foot.L", 0.0f},
                       }},
        {"Marker_L16", {
                           {"foot.L", 0.0f},
                       }},
        {"Marker_L17", {
                           {"foot.L", 0.0f},
                       }},
        {"Marker_L18", {
                           {"lower_leg.L", 1.0f},
                           {"foot.L", 0.0f},
                       }},
        {"Marker_L19", {
                           {"hand.L", 0.0f},
                       }},
        {"Marker_L20", {
                           {"hand.L", 0.0f},
                       }},

        {"Marker_F1", {
                          {"Chest", 0.0f},
                      }},
        {"Marker_B1", {
                          {"Chest", 0.0f},
                      }},
        {"Marker_F2", {
                          {"Spine", 1.0f},
                          {"Chest", 0.0f},
                      }},
        {"Marker_B2", {
                          {"Spine", 1.0f},
                          {"Chest", 0.0f},
                      }},

        {"Marker_H1", {
                          {"Neck", 0.0f},
                      }},
        {"Marker_H2", {
                          {"Neck", 0.0f},
                      }},
        {"Marker_H3", {
                          {"Neck", 0.0f},
                      }},
        {"Marker_H4", {
                          {"Neck", 0.0f},
                      }},
        {"Marker_H5", {
                          {"Neck", 0.0f},
                      }},
    };

    std::map<std::string, std::string> anchor = {
        {"upper_arm.R", "Marker_R0"},
        {"lower_arm.R", "Marker_R3"},
        {"hand.R", "Marker_R6"},
        {"upper_arm.L", "Marker_L0"},
        {"lower_arm.L", "Marker_L3"},
        {"hand.L", "Marker_L6"},

        {"upper_leg.R", "Marker_R9"},
        {"lower_leg.R", "Marker_R11"},
        {"foot.R", "Marker_R14"},
        {"upper_leg.L", "Marker_L9"},
        {"lower_leg.L", "Marker_L11"},
        {"foot.L", "Marker_L14"},

        {"Spine", "Marker_F2"},
        {"Chest", "Marker_F2"},
        {"Neck", "Marker_H1"},
    };

    std::vector<json> j_objs;
    for (const auto &mesh : meshs)
    {
        json j_obj;

        const auto name = mesh->parent.lock()->name;
        j_obj["name"] = name;
        j_obj["position"] = vec3_to_json(glm::vec3(mesh->calculate_absolute_transform()[3]));
        j_obj["weights"] = weights[name];
        if (name.find("Cube") == 0)
        {
            j_obj["pose"] = mat4_to_json(mesh->calculate_absolute_transform());
        }
        j_objs.push_back(j_obj);
    }

    std::sort(j_objs.begin(), j_objs.end(), [](const json &a, const json &b)
              { return a["name"].get<std::string>() < b["name"].get<std::string>(); });

    std::vector<json> j_bones;
    for (const auto &skeleton : skeletons)
    {
        json j_bone;
        j_bone["name"] = skeleton->name;
        j_bone["transform"] = mat4_to_json(skeleton->parent.lock()->calculate_absolute_transform());
        j_bone["anchor"] = anchor[skeleton->name];
        j_bone["length"] = glm::length(glm::vec3(skeleton->transform[3]));
        j_bones.push_back(j_bone);
    }

    json j = {
        {"objects", j_objs},
        {"bones", j_bones},
    };

    std::ofstream ofs;
    ofs.open(output_name, std::ios::out | std::ios::binary);
    ofs << j.dump(2);

    std::cout << "Successfully converted." << std::endl;

    return 0;
}

int main(int argc, const char *argv[])
{
#if 0
    namespace po = boost::program_options;

    po::options_description desc("Options");
    desc.add_options()("model", "Convert model")("tpose", "Convert tpose");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help"))
    {
        std::cout << desc << "\n";
        return -1;
    }
#endif
    const auto model = true;
    const auto tpose = false;
    if (tpose)
    {
        convert_tpose(argc, argv);
    }
    else if (model)
    {
        convert_model(argc, argv);
    }

    return 0;
}