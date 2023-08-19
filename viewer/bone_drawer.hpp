#pragma once

#include <vector>
#include <cmath>
#include <glm/glm.hpp>

class bone_drawer
{
    int shader;

public:
    bone_drawer(const glm::u8vec4 color = glm::u8vec4(200, 200, 200, 255));
    ~bone_drawer() {}

    void initialize();

    void draw(glm::mat4 wvp) const;

protected:
private:
    void add_vertex(float x, float y, float z);
    void add_color(float r, float g, float b, float a);

    std::vector<float> vertices;
    std::vector<float> colors;
    std::vector<unsigned int> indices;

    unsigned int vertex_buffer;
    unsigned int color_buffer;
    unsigned int index_buffer;

    unsigned int vao;
};
