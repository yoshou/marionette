#include "box_drawer.hpp"

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <array>
#include <fstream>
#include <iostream>
#include <string>

#include "shader_utils.hpp"

box_drawer::box_drawer() {
  std::array<glm::vec4, 8> points = {{{-1.f, -1.f, -1.f, 1.f},
                                      {1.f, -1.f, -1.f, 1.f},
                                      {-1.f, 1.f, -1.f, 1.f},
                                      {1.f, 1.f, -1.f, 1.f},
                                      {-1.f, -1.f, 1.f, 1.f},
                                      {1.f, -1.f, 1.f, 1.f},
                                      {-1.f, 1.f, 1.f, 1.f},
                                      {1.f, 1.f, 1.f, 1.f}}};

  glm::u8vec4 color(200, 200, 200, 255);
  for (std::size_t i = 0; i < points.size(); i++) {
    add_vertex(points[i].x, points[i].y, points[i].z);
    add_color(color.r, color.g, color.b, color.a);
  }

  // back
  {
    indices.push_back(0);
    indices.push_back(2);
    indices.push_back(3);
  }
  {
    indices.push_back(0);
    indices.push_back(3);
    indices.push_back(1);
  }

  // front
  {
    indices.push_back(4);
    indices.push_back(5);
    indices.push_back(7);
  }
  {
    indices.push_back(4);
    indices.push_back(7);
    indices.push_back(6);
  }

  // bottom
  {
    indices.push_back(0);
    indices.push_back(1);
    indices.push_back(5);
  }
  {
    indices.push_back(0);
    indices.push_back(5);
    indices.push_back(4);
  }

  // top
  {
    indices.push_back(2);
    indices.push_back(6);
    indices.push_back(7);
  }
  {
    indices.push_back(2);
    indices.push_back(7);
    indices.push_back(3);
  }

  // left
  {
    indices.push_back(0);
    indices.push_back(4);
    indices.push_back(6);
  }
  {
    indices.push_back(0);
    indices.push_back(6);
    indices.push_back(2);
  }

  // right
  {
    indices.push_back(1);
    indices.push_back(3);
    indices.push_back(7);
  }
  {
    indices.push_back(1);
    indices.push_back(7);
    indices.push_back(5);
  }
}

void box_drawer::add_vertex(float x, float y, float z) {
  vertices.push_back(x);
  vertices.push_back(y);
  vertices.push_back(z);
}
void box_drawer::add_color(float r, float g, float b, float a) {
  colors.push_back(r);
  colors.push_back(g);
  colors.push_back(b);
  colors.push_back(a);
}

void box_drawer::draw(glm::mat4 wvp) const {
  glUseProgram(shader);
  glUniformMatrix4fv(glGetUniformLocation(shader, "pvw"), 1, GL_FALSE, &wvp[0][0]);

  glBindVertexArray(vao);
  glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  glUseProgram(0);
}

void box_drawer::initialize() {
  shader =
      shader_utils::load_program("../viewer/shaders/color.vert", "../viewer/shaders/color.frag");

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
