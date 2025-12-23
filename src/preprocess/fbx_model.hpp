#pragma once

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <glm/glm.hpp>

namespace marionette::model
{
    class node_t
    {
    public:
        std::string name;
        glm::mat4 transform;
        std::weak_ptr<node_t> parent;
        std::vector<std::shared_ptr<node_t>> children;

        glm::mat4 calculate_absolute_transform() const
        {
            if (auto parent_ptr = parent.lock())
            {
                return parent_ptr->calculate_absolute_transform() * transform;
            }

            return transform;
        }

        node_t()
            : name(), transform(1.0f), parent(), children()
        {
        }

        virtual ~node_t() = default;
    };

    struct bounding_box_t
    {
        glm::vec3 center;
        glm::vec3 half;

        bounding_box_t()
            : center(0.0f), half(0.0f)
        {
        }
    };

    class mesh_node_t : public node_t
    {
    public:
        bounding_box_t bounding_box;
        std::map<std::string, float> weights;
    };

    class skeleton_node_t : public node_t
    {
    public:
    };

    template <typename Func>
    static void traverse_node(const std::shared_ptr<node_t> &node, Func pred)
    {
        pred(node);

        for (const auto &child : node->children)
        {
            traverse_node(child, pred);
        }
    }
}
