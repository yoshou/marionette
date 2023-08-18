#pragma once

#include <nlohmann/json.hpp>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

namespace glm
{
    static void to_json(nlohmann::json &j, const glm::vec2 &v)
    {
        j = {v.x, v.y};
    };
    static void from_json(const nlohmann::json &j, glm::vec2 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::vec3 &v)
    {
        j = {v.x, v.y, v.z};
    };
    static void from_json(const nlohmann::json &j, glm::vec3 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
        v.z = j[2].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::vec4 &v)
    {
        j = {v.x, v.y, v.z, v.w};
    };
    static void from_json(const nlohmann::json &j, glm::vec4 &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
        v.z = j[2].get<float>();
        v.w = j[3].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::quat &v)
    {
        j = {v.x, v.y, v.z, v.w};
    };
    static void from_json(const nlohmann::json &j, glm::quat &v)
    {
        v.x = j[0].get<float>();
        v.y = j[1].get<float>();
        v.z = j[2].get<float>();
        v.w = j[3].get<float>();
    }
    static void to_json(nlohmann::json &j, const glm::mat3 &m)
    {
        std::array<float, 9> values;
        for (std::size_t i = 0; i < 3; i++)
        {
            for (std::size_t j = 0; j < 3; j++)
            {
                values[i * 3 + j] = m[i][j];
            }
        }
        j = values;
    };
    static void from_json(const nlohmann::json &j, glm::mat3 &m)
    {
        std::array<float, 9> values;
        for (std::size_t i = 0; i < 3; i++)
        {
            for (std::size_t k = 0; k < 3; k++)
            {
                m[i][k] = j[i * 3 + k].get<float>();
            }
        }
    }
    static void to_json(nlohmann::json &j, const glm::mat4 &m)
    {
        std::array<float, 16> values;
        for (std::size_t i = 0; i < 4; i++)
        {
            for (std::size_t j = 0; j < 4; j++)
            {
                values[i * 4 + j] = m[i][j];
            }
        }
        j = values;
    };
    static void from_json(const nlohmann::json &j, glm::mat4 &m)
    {
        std::array<float, 16> values;
        for (std::size_t i = 0; i < 4; i++)
        {
            for (std::size_t k = 0; k < 4; k++)
            {
                m[i][k] = j[i * 4 + k].get<float>();
            }
        }
    }
}
