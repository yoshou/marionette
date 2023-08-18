#pragma once

#include <vector>
#include <cmath>
#include <glm/glm.hpp>

class drawer2d
{
    int shader;

public:
    drawer2d();
    ~drawer2d() {}

    void initialize();

    void draw_rect(const glm::vec2& position, const glm::vec2& size, const glm::vec4 &color) const;

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
};