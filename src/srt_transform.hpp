#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/matrix_decompose.hpp>

template <typename T>
struct srt_transform
{
    using value_type = T;
    using vec3 = glm::vec<3, value_type, glm::defaultp>;
    using vec4 = glm::vec<4, value_type, glm::defaultp>;
    using quat = glm::qua<value_type, glm::defaultp>;
    using mat4 = glm::mat<4, 4, value_type, glm::defaultp>;
public:
    vec3 scale;
    quat rotation;
    vec3 translation;

public:
    inline srt_transform()
        : scale(T{1}), rotation(T{1}, T{0}, T{0}, T{0}), translation(T{0})
    {
    }

    inline srt_transform(const vec3 scale, const quat rotation, const vec3 translation)
        : scale(scale), rotation(rotation), translation(translation)
    {
    }

    inline srt_transform(const srt_transform &other)
        : scale(other.scale), rotation(other.rotation), translation(other.translation)
    {
    }

    srt_transform& operator=(const srt_transform &other)
    {
        scale = other.scale;
        rotation = other.rotation;
        translation = other.translation;
        return *this;
    }

    inline static const srt_transform &identity()
    {
        static const srt_transform identity;
        return identity;
    }

    static inline srt_transform from_matrix(const mat4 &matrix)
    {
        vec3 scale;
        quat rotation;
        vec3 translation;
        vec3 skew;
        vec4 perspective;
        glm::decompose(matrix, scale, rotation, translation, skew, perspective);
        
        return sqt_transform(scale, rotation, translation);
    }

    inline static srt_transform lerp(const srt_transform &s1, const srt_transform &s2, float amount)
    {
        srt_transform result;
        result.scale = glm::lerp(s1.scale, s2.scale, amount);
        result.translation = glm::lerp(s1.translation, s2.translation, amount);
        result.rotation = glm::slerp(s1.rotation, s2.rotation, amount);

        return result;
    }

    inline mat4 create_matrix()
    {
        return glm::translate(translation) * glm::toMat4(rotation) * glm::scale(scale);
    }
};

template <typename T>
static inline srt_transform<T> operator*(const srt_transform<T> &b, const srt_transform<T> &a)
{
    srt_transform<T> c;

    c.translation = glm::rotate(b.rotation, a.translation * b.scale) + b.translation;
    c.rotation = b.rotation * a.rotation;
    c.scale = b.scale * a.scale;

    return c;
}
