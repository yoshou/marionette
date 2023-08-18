#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/transform.hpp>

struct line3
{
    glm::vec3 origin;
    glm::vec3 direction;
};

static line3 to_line(const glm::mat4& m)
{
    return line3{glm::vec3(m[3]), glm::normalize(glm::mat3(m) * glm::vec3(0.f, 1.f, 0.f))};
}

static inline glm::vec3 twist(const weighted_point &point, const glm::mat4 &pose, const glm::mat4 &inv_pose, float angle)
{
    const auto weight = point.weight;
    const auto rotation = glm::slerp(glm::quat(glm::vec3(0.f)), glm::angleAxis(glm::radians(angle), glm::vec3(0.f, 1.f, 0.f)), weight);
    const auto pos_in_bone = inv_pose * glm::vec4(point.position, 1.f);
    const auto twisted_pos = rotation * glm::vec3(pos_in_bone);

    return glm::vec3(pose * glm::vec4(twisted_pos, 1.f));
}

static inline glm::vec3 twist(const weighted_point &point, const line3 &twist_axis, float angle)
{
    const auto weight = point.weight;
    const auto twist_quat = glm::angleAxis(glm::radians(angle) * weight, twist_axis.direction);
    const auto twisted_p = (twist_quat * (point.position - twist_axis.origin)) + twist_axis.origin;

    return twisted_p;
}

static inline glm::vec3 rotate_local(const glm::vec3 &point, const glm::mat4 &pose, const glm::mat3 rotation)
{
    const auto pos_in_bone = glm::inverse(pose) * glm::vec4(point, 1.f);
    const auto rotated_pos = rotation * glm::vec3(pos_in_bone);

    return glm::vec3(pose * glm::vec4(rotated_pos, 1.f));
}

template <typename T>
static glm::vec3 scaling(const glm::vec3 &p, const glm::vec3 &origin, const T *scale)
{
    glm::vec3 scale_vec(static_cast<float>(scale[0]),
                        static_cast<float>(scale[1]),
                        static_cast<float>(scale[2]));

    return (p - origin) * scale_vec + origin;
}

template <typename T>
static glm::vec3 transform_axis_angle(const glm::vec3 &p, const T *axis_angle, const T *translation)
{
    glm::vec3 axis_angle_vec(static_cast<float>(axis_angle[0]),
                             static_cast<float>(axis_angle[1]),
                             static_cast<float>(axis_angle[2]));

    glm::vec3 trans_vec(static_cast<float>(translation[0]),
                        static_cast<float>(translation[1]),
                        static_cast<float>(translation[2]));

    const auto angle = glm::length(axis_angle_vec);

    if (angle > std::numeric_limits<float>::epsilon())
    {
        const auto axis = glm::normalize(axis_angle_vec);
        const auto quat = glm::angleAxis(angle, axis);
        return quat * p + trans_vec;
    }
    else
    {
        return p + trans_vec;
    }
}

template <typename T>
static glm::vec3 transform_twist_axis_angle(const glm::vec3 &p, const T twist_angle, const line3 &twist_axis, const float twist_weight, const T *axis_angle, const T *translation)
{
    const auto twist_quat = glm::angleAxis(static_cast<float>(twist_angle) * twist_weight, glm::normalize(twist_axis.direction));
    const auto twisted_p = (twist_quat * (p - twist_axis.origin)) + twist_axis.origin;

    return transform_axis_angle(twisted_p, axis_angle, translation);
}

template <typename T>
static glm::vec3 transform_quat(const glm::vec3 &p, const T *rotation, const T *translation)
{
    const glm::quat quat(static_cast<float>(rotation[0]),
                         static_cast<float>(rotation[1]),
                         static_cast<float>(rotation[2]),
                         static_cast<float>(rotation[3]));

    const glm::vec3 trans(static_cast<float>(translation[0]),
                          static_cast<float>(translation[1]),
                          static_cast<float>(translation[2]));

    return quat * p + trans;
}

template <typename T>
static glm::vec3 transform_twist_quat(const glm::vec3 &p, const T twist_angle, const line3 &twist_axis, const float twist_weight, const T *rotation, const T *translation)
{
    const auto twist_quat = glm::angleAxis(static_cast<float>(twist_angle) * twist_weight, glm::normalize(twist_axis.direction));
    const auto twisted_p = (twist_quat * (p - twist_axis.origin)) + twist_axis.origin;

    return transform_quat(twisted_p, rotation, translation);
}

template <typename T>
static line3 transform_axis_angle(const line3 &l, const T *axis_angle, const T *translation)
{
    glm::vec3 axis_angle_vec(static_cast<float>(axis_angle[0]),
                             static_cast<float>(axis_angle[1]),
                             static_cast<float>(axis_angle[2]));

    glm::vec3 trans_vec(static_cast<float>(translation[0]),
                        static_cast<float>(translation[1]),
                        static_cast<float>(translation[2]));

    const auto angle = glm::length(axis_angle_vec);

    if (angle > std::numeric_limits<float>::epsilon())
    {
        const auto axis = glm::normalize(axis_angle_vec);
        const auto quat = glm::angleAxis(angle, axis);
        return line3{ 
            quat * l.origin + trans_vec,
            glm::normalize(quat * l.direction)
        };
    }
    else
    {
        return line3{
            l.origin + trans_vec,
            glm::normalize(l.direction)
        };
    }
}

template <typename T>
static line3 transform_quat(const line3 &l, const T *rotation, const T *translation)
{
    const glm::quat quat(static_cast<float>(rotation[0]),
                       static_cast<float>(rotation[1]),
                       static_cast<float>(rotation[2]),
                       static_cast<float>(rotation[3]));

    const glm::vec3 trans(static_cast<float>(translation[0]),
                        static_cast<float>(translation[1]),
                        static_cast<float>(translation[2]));

    return line3{
        quat * l.origin + trans,
        glm::normalize(quat * l.direction)};
}

static inline glm::vec3 transform_coordinate(glm::vec3 position, const glm::mat4 &transform)
{
    return glm::vec3(transform * glm::vec4(position, 1.f));
}

static inline weighted_point transform_coordinate(const weighted_point &point, const glm::mat4 &transform)
{
    return weighted_point{glm::vec3(transform * glm::vec4(point.position, 1.f)), point.weight, point.id};
}

template <typename T>
static inline std::vector<T> transform_coordinate(const std::vector<T> &positions, const glm::mat4 &transform)
{
    std::vector<T> result;
    for (const auto p : positions)
    {
        result.push_back(transform_coordinate(p, transform));
    }
    return result;
}
