#define _USE_MATH_DEFINES

#include "fbx_loader.hpp"

#include <iostream>
#include <glm/gtx/transform.hpp>

#include <fbxsdk.h>

#define fbxsdk FBXSDK_NAMESPACE

namespace marionette::model::fbx
{
    template <typename Func>
    static void convert_node(const fbxsdk::FbxNode *node, const std::shared_ptr<node_t> &parent, Func pred)
    {
        const auto processed = pred(node);
        parent->children.push_back(processed);
        processed->parent = parent;

        const auto num_children = node->GetChildCount();
        for (int i = 0; i < num_children; i++)
        {
            const auto child = node->GetChild(i);
            convert_node(child, processed, pred);
        }
    }

    template <typename Func>
    static std::shared_ptr<node_t> convert_node(const fbxsdk::FbxNode *node, Func pred)
    {
        const auto root = std::make_shared<node_t>();
        convert_node(node, root, pred);
        root->children[0]->parent.reset();
        return root->children[0];
    }

    template <typename T>
    struct fbx_destroy_deleter
    {
        void operator()(T *obj)
        {
            if (obj)
            {
                obj->Destroy();
            }
        }
    };

    template <typename T, typename... Args>
    static std::shared_ptr<T> create_fbx_object(Args... args)
    {
        return std::shared_ptr<T>(T::Create(args...), fbx_destroy_deleter<T>());
    }

    template <typename T>
    static T deg_to_rad(T value)
    {
        return static_cast<T>(static_cast<double>(value) * M_PI / 180.0);
    }

    static glm::mat4 calculate_node_transform(const fbxsdk::FbxNode *node)
    {
        const auto scale = node->LclScaling.Get();
        const auto translation = node->LclTranslation.Get();
        const auto rotation = node->LclRotation.Get();

        const auto rotate_x = glm::rotate(deg_to_rad(static_cast<float>(rotation[0])), glm::vec3(1, 0, 0));
        const auto rotate_y = glm::rotate(deg_to_rad(static_cast<float>(rotation[1])), glm::vec3(0, 1, 0));
        const auto rotate_z = glm::rotate(deg_to_rad(static_cast<float>(rotation[2])), glm::vec3(0, 0, 1));

        glm::mat4 transform = glm::translate(glm::vec3(translation[0], translation[1], translation[2])) *
                              rotate_z * rotate_y * rotate_x *
                              glm::scale(glm::vec3(scale[0], scale[1], scale[2]));
        return transform;
    }

    static std::shared_ptr<node_t> process_mesh(const fbxsdk::FbxMesh *mesh)
    {
        const auto result = std::make_shared<mesh_node_t>();
        result->transform = glm::mat4(1.0f);
        result->name = std::string(mesh->GetName());

        int num_deformers = mesh->GetDeformerCount();
        for (int i = 0; i < num_deformers; i++)
        {
            const auto deformer = mesh->GetDeformer(i, nullptr);
            if (const auto skin = FbxCast<FbxSkin>(deformer))
            {
                int num_clusters = skin->GetClusterCount();
                for (int j = 0; j < num_clusters; j++)
                {
                    const auto cluster = skin->GetCluster(j);
                    const auto num_cp = cluster->GetControlPointIndicesCount();
                    if (num_cp == 0)
                    {
                        continue;
                    }
                    const auto cp_weights = cluster->GetControlPointWeights();
                    result->weights.insert(std::make_pair(cluster->GetLink()->GetName(), cp_weights[0]));
                }
            }
        }
        return result;
    }

    static std::shared_ptr<node_t> process_skeleton(const fbxsdk::FbxSkeleton *skeleton)
    {
        const auto result = std::make_shared<skeleton_node_t>();
        result->transform = glm::mat4(1.0f);
        result->name = std::string(skeleton->GetName());
        return result;
    }

    static std::shared_ptr<node_t> process_node_attribute(const fbxsdk::FbxNodeAttribute *attrib)
    {
        if (const auto skeleton = FbxCast<FbxSkeleton>(attrib))
        {
            return process_skeleton(skeleton);
        }
        else if (const auto mesh = FbxCast<FbxMesh>(attrib))
        {
            return process_mesh(mesh);
        }
        else
        {
            const auto result = std::make_shared<node_t>();
            return result;
        }
    }

    static std::shared_ptr<node_t> process_node(const fbxsdk::FbxNode *node)
    {
        const auto result = std::make_shared<node_t>();
        result->transform = calculate_node_transform(node);
        result->name = std::string(node->GetName());
        for (int i = 0; i < node->GetNodeAttributeCount(); i++)
        {
            const auto result_attrib = process_node_attribute(node->GetNodeAttributeByIndex(i));
            result_attrib->parent = result;
            result->children.push_back(result_attrib);
        }
        return result;
    }

    std::shared_ptr<node_t> load_model(const std::string &filename)
    {
        auto manager = create_fbx_object<FbxManager>();
        auto importer = create_fbx_object<FbxImporter>(manager.get(), "untitled");

        if (!importer->Initialize(filename.c_str()))
        {
            std::cout << "Failed to open the FBX file." << std::endl;
            exit(-1);
        }

        auto scene = create_fbx_object<FbxScene>(manager.get(), "untitled");
        if (!importer->Import(scene.get()))
        {
            std::cout << "Failed to import the scene." << std::endl;
            exit(-1);
        }

        FbxAxisSystem(FbxAxisSystem::eOpenGL).ConvertScene(scene.get());
        const auto root_node = scene->GetRootNode();

        const auto processed_node = convert_node(root_node, process_node);

        return processed_node;
    }
}
