#include "drawer2d.hpp"

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <fstream>
#include <glm/gtx/transform.hpp>
#include <iostream>
#include <string>

#include "grid_drawer.hpp"
#include "shader_utils.hpp"

drawer2d::drawer2d() {
  {
    glm::vec3 v0(0.0f, 0.0f, -1.0f);
    glm::vec3 v1(1.0f, 0.0f, -1.0f);
    glm::vec3 v2(1.0f, 1.0f, -1.0f);
    glm::vec3 v3(0.0f, 1.0f, -1.0f);
    glm::vec4 color(1, 1, 1, 1);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v0.x, v0.y, v0.z);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v1.x, v1.y, v1.z);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v1.x, v1.y, v1.z);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v2.x, v2.y, v2.z);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v2.x, v2.y, v2.z);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v3.x, v3.y, v3.z);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v3.x, v3.y, v3.z);
    add_color(color.r, color.g, color.b, color.a);
    add_vertex(v0.x, v0.y, v0.z);
  }
}

void drawer2d::add_vertex(float x, float y, float z) {
  vertices.push_back(x);
  vertices.push_back(y);
  vertices.push_back(z);

  indices.push_back(static_cast<unsigned int>(indices.size()));
}
void drawer2d::add_color(float r, float g, float b, float a) {
  colors.push_back(r);
  colors.push_back(g);
  colors.push_back(b);
  colors.push_back(a);
}

void drawer2d::draw_rect(const glm::vec2 &position, const glm::vec2 &size,
                         const glm::vec4 &color) const {
  glm::mat4 wvp = glm::translate(glm::vec3(-1.0f, 1.0f, 0.0f)) *
                  glm::scale(glm::vec3(2.0f, -2.0f, 0.0f)) *
                  glm::translate(glm::vec3(position.x, position.y, 0.0f)) *
                  glm::scale(glm::vec3(size.x, size.y, 0.0f));

  glUseProgram(shader);
  glUniformMatrix4fv(glGetUniformLocation(shader, "pvw"), 1, GL_FALSE, &wvp[0][0]);

  // set line colour
  float values[] = {(float)color.x, (float)color.y, (float)color.z, (float)color.w};
  glUniform4fv(glGetUniformLocation(shader, "color"), 1, values);

  glBindVertexArray(vao);
  glDrawElements(GL_LINES, (unsigned int)indices.size(), GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  glUseProgram(0);
}

void drawer2d::initialize() {
  shader = shader_utils::load_program("../viewer/shaders/position.vert",
                                      "../viewer/shaders/position.frag");

  glGenBuffers(1, &vertex_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &color_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), colors.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &index_buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(),
               GL_STATIC_DRAW);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
  glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(1);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

  glBindVertexArray(0);
}
