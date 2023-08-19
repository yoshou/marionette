#pragma once

#include <vector>
#include <cmath>
#include <glm/glm.hpp>

class grid_drawer
{
    int shader;

public:
    grid_drawer();
    ~grid_drawer() {}

    void initialize();

    void draw(glm::mat4 wvp) const;

protected:
private:
    void addVertex(float x, float y, float z);
    void addColor(float r, float g, float b, float a);

    std::vector<float> vertices;
    std::vector<float> colors;
    std::vector<unsigned int> indices;

    unsigned int vertex_buffer;
    unsigned int color_buffer;
    unsigned int index_buffer;

    unsigned int vao;
};
