#include "sphere_drawer.hpp"

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <fstream>
#include <iostream>
#include <string>

#include "shader_utils.hpp"

sphere_drawer::sphere_drawer(int sectors, int stacks, bool smooth) : interleaved_stride(32) {
  set(sectors, stacks, smooth);
}

void sphere_drawer::set(int sectors, int stacks, bool smooth) {
  this->sector_count = sectors;
  if (sectors < MIN_SECTOR_COUNT) this->sector_count = MIN_SECTOR_COUNT;
  this->stack_count = stacks;
  if (sectors < MIN_STACK_COUNT) this->sector_count = MIN_STACK_COUNT;
  this->smooth = smooth;

  if (smooth)
    build_vertices_smooth();
  else
    build_vertices_flat();
}

void sphere_drawer::set_sector_count(int sectors) {
  if (sectors != this->sector_count) set(sectors, stack_count, smooth);
}

void sphere_drawer::set_stack_count(int stacks) {
  if (stacks != this->stack_count) set(sector_count, stacks, smooth);
}

void sphere_drawer::set_smooth(bool smooth) {
  if (this->smooth == smooth) return;

  this->smooth = smooth;
  if (smooth)
    build_vertices_smooth();
  else
    build_vertices_flat();
}

void sphere_drawer::draw(glm::mat4 wvp, const float color[4]) const {
  glUseProgram(shader);
  glUniformMatrix4fv(glGetUniformLocation(shader, "pvw"), 1, GL_FALSE, &wvp[0][0]);

  // set line colour
  float values[] = {(float)color[0], (float)color[1], (float)color[2], (float)color[3]};
  glUniform4fv(glGetUniformLocation(shader, "color"), 1, values);

  glBindVertexArray(vao);
  glDrawElements(GL_TRIANGLES, (unsigned int)indices.size(), GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  glUseProgram(0);
}

void sphere_drawer::draw_lines(glm::mat4 wvp, const float color[4]) const {
  glUseProgram(shader);
  glUniformMatrix4fv(glGetUniformLocation(shader, "pvw"), 1, GL_FALSE, &wvp[0][0]);

  // set line colour
  float values[] = {(float)color[0], (float)color[1], (float)color[2], (float)color[3]};
  glUniform4fv(glGetUniformLocation(shader, "color"), 1, values);

  glBindVertexArray(line_vao);
  glDrawElements(GL_LINES, (unsigned int)line_indices.size(), GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);

  glUseProgram(0);
}

void sphere_drawer::draw_with_lines(glm::mat4 wvp, const float line_color[4]) const {
  this->draw(wvp, line_color);

  // draw lines with VA
  draw_lines(wvp, line_color);
}

void sphere_drawer::clear_arrays() {
  std::vector<float>().swap(vertices);
  std::vector<float>().swap(normals);
  std::vector<float>().swap(tex_coords);
  std::vector<unsigned int>().swap(indices);
  std::vector<unsigned int>().swap(line_indices);
}

void sphere_drawer::initialize() {
  shader = shader_utils::load_program("../viewer/shaders/position.vert",
                                      "../viewer/shaders/position.frag");

  glGenBuffers(1, &vertex_buffer);
  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

  glGenBuffers(1, &index_buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(),
               GL_STATIC_DRAW);

  glGenBuffers(1, &line_index_buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, line_index_buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, line_indices.size() * sizeof(unsigned int),
               line_indices.data(), GL_STATIC_DRAW);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);

  glBindVertexArray(0);

  glGenVertexArrays(1, &line_vao);
  glBindVertexArray(line_vao);

  glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
  glEnableVertexAttribArray(0);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, line_index_buffer);

  glBindVertexArray(0);
}

void sphere_drawer::build_vertices_smooth() {
  const float PI = static_cast<float>(acos(-1));

  // clear memory of prev arrays
  clear_arrays();

  float x, y, z, xy;  // vertex position
  float nx, ny, nz;   // normal
  float s, t;         // texCoord

  float sector_step = 2 * PI / sector_count;
  float stack_step = PI / stack_count;
  float sector_angle, stack_angle;

  for (int i = 0; i <= stack_count; ++i) {
    stack_angle = PI / 2 - i * stack_step;  // starting from pi/2 to -pi/2
    xy = cosf(stack_angle);                 // r * cos(u)
    z = sinf(stack_angle);                  // r * sin(u)

    // add (sector_count+1) vertices per stack
    // the first and last vertices have same position and normal, but different tex coords
    for (int j = 0; j <= sector_count; ++j) {
      sector_angle = j * sector_step;  // starting from 0 to 2pi

      // vertex position
      x = xy * cosf(sector_angle);  // r * cos(u) * cos(v)
      y = xy * sinf(sector_angle);  // r * cos(u) * sin(v)
      add_vertex(x, y, z);

      // normalized vertex normal
      nx = x;
      ny = y;
      nz = z;
      add_normal(nx, ny, nz);

      // vertex tex coord between [0, 1]
      s = (float)j / sector_count;
      t = (float)i / stack_count;
      add_tex_coord(s, t);
    }
  }

  // indices
  //  k1--k1+1
  //  |  / |
  //  | /  |
  //  k2--k2+1
  unsigned int k1, k2;
  for (int i = 0; i < stack_count; ++i) {
    k1 = i * (sector_count + 1);  // beginning of current stack
    k2 = k1 + sector_count + 1;   // beginning of next stack

    for (int j = 0; j < sector_count; ++j, ++k1, ++k2) {
      // 2 triangles per sector excluding 1st and last stacks
      if (i != 0) {
        add_indices(k1, k2, k1 + 1);  // k1---k2---k1+1
      }

      if (i != (stack_count - 1)) {
        add_indices(k1 + 1, k2, k2 + 1);  // k1+1---k2---k2+1
      }

      // vertical lines for all stacks
      line_indices.push_back(k1);
      line_indices.push_back(k2);
      if (i != 0)  // horizontal lines except 1st stack
      {
        line_indices.push_back(k1);
        line_indices.push_back(k1 + 1);
      }
    }
  }

  // generate interleaved vertex array as well
  build_interleaved_vertices();
}

void sphere_drawer::build_vertices_flat() {
  const float PI = static_cast<float>(acos(-1));

  // tmp vertex definition (x,y,z,s,t)
  struct Vertex {
    float x, y, z, s, t;
  };
  std::vector<Vertex> tmp_vertices;

  float sector_step = 2 * PI / sector_count;
  float stack_step = PI / stack_count;
  float sector_angle, stack_angle;

  // compute all vertices first, each vertex contains (x,y,z,s,t) except normal
  for (int i = 0; i <= stack_count; ++i) {
    stack_angle = PI / 2 - i * stack_step;  // starting from pi/2 to -pi/2
    float xy = cosf(stack_angle);           // r * cos(u)
    float z = sinf(stack_angle);            // r * sin(u)

    // add (sector_count+1) vertices per stack
    // the first and last vertices have same position and normal, but different tex coords
    for (int j = 0; j <= sector_count; ++j) {
      sector_angle = j * sector_step;  // starting from 0 to 2pi

      Vertex vertex;
      vertex.x = xy * cosf(sector_angle);  // x = r * cos(u) * cos(v)
      vertex.y = xy * sinf(sector_angle);  // y = r * cos(u) * sin(v)
      vertex.z = z;                        // z = r * sin(u)
      vertex.s = (float)j / sector_count;  // s
      vertex.t = (float)i / stack_count;   // t
      tmp_vertices.push_back(vertex);
    }
  }

  // clear memory of prev arrays
  clear_arrays();

  Vertex v1, v2, v3, v4;  // 4 vertex positions and tex coords
  std::vector<float> n;   // 1 face normal

  int i, j, k, vi1, vi2;
  int index = 0;  // index for vertex
  for (i = 0; i < stack_count; ++i) {
    vi1 = i * (sector_count + 1);  // index of tmp_vertices
    vi2 = (i + 1) * (sector_count + 1);

    for (j = 0; j < sector_count; ++j, ++vi1, ++vi2) {
      // get 4 vertices per sector
      //  v1--v3
      //  |    |
      //  v2--v4
      v1 = tmp_vertices[vi1];
      v2 = tmp_vertices[vi2];
      v3 = tmp_vertices[vi1 + 1];
      v4 = tmp_vertices[vi2 + 1];

      // if 1st stack and last stack, store only 1 triangle per sector
      // otherwise, store 2 triangles (quad) per sector
      if (i == 0)  // a triangle for first stack ==========================
      {
        // put a triangle
        add_vertex(v1.x, v1.y, v1.z);
        add_vertex(v2.x, v2.y, v2.z);
        add_vertex(v4.x, v4.y, v4.z);

        // put tex coords of triangle
        add_tex_coord(v1.s, v1.t);
        add_tex_coord(v2.s, v2.t);
        add_tex_coord(v4.s, v4.t);

        // put normal
        n = compute_face_normal(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v4.x, v4.y, v4.z);
        for (k = 0; k < 3; ++k)  // same normals for 3 vertices
        {
          add_normal(n[0], n[1], n[2]);
        }

        // put indices of 1 triangle
        add_indices(index, index + 1, index + 2);

        // indices for line (first stack requires only vertical line)
        line_indices.push_back(index);
        line_indices.push_back(index + 1);

        index += 3;                       // for next
      } else if (i == (stack_count - 1))  // a triangle for last stack =========
      {
        // put a triangle
        add_vertex(v1.x, v1.y, v1.z);
        add_vertex(v2.x, v2.y, v2.z);
        add_vertex(v3.x, v3.y, v3.z);

        // put tex coords of triangle
        add_tex_coord(v1.s, v1.t);
        add_tex_coord(v2.s, v2.t);
        add_tex_coord(v3.s, v3.t);

        // put normal
        n = compute_face_normal(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
        for (k = 0; k < 3; ++k)  // same normals for 3 vertices
        {
          add_normal(n[0], n[1], n[2]);
        }

        // put indices of 1 triangle
        add_indices(index, index + 1, index + 2);

        // indices for lines (last stack requires both vert/hori lines)
        line_indices.push_back(index);
        line_indices.push_back(index + 1);
        line_indices.push_back(index);
        line_indices.push_back(index + 2);

        index += 3;  // for next
      } else         // 2 triangles for others ====================================
      {
        // put quad vertices: v1-v2-v3-v4
        add_vertex(v1.x, v1.y, v1.z);
        add_vertex(v2.x, v2.y, v2.z);
        add_vertex(v3.x, v3.y, v3.z);
        add_vertex(v4.x, v4.y, v4.z);

        // put tex coords of quad
        add_tex_coord(v1.s, v1.t);
        add_tex_coord(v2.s, v2.t);
        add_tex_coord(v3.s, v3.t);
        add_tex_coord(v4.s, v4.t);

        // put normal
        n = compute_face_normal(v1.x, v1.y, v1.z, v2.x, v2.y, v2.z, v3.x, v3.y, v3.z);
        for (k = 0; k < 4; ++k)  // same normals for 4 vertices
        {
          add_normal(n[0], n[1], n[2]);
        }

        // put indices of quad (2 triangles)
        add_indices(index, index + 1, index + 2);
        add_indices(index + 2, index + 1, index + 3);

        // indices for lines
        line_indices.push_back(index);
        line_indices.push_back(index + 1);
        line_indices.push_back(index);
        line_indices.push_back(index + 2);

        index += 4;  // for next
      }
    }
  }

  // generate interleaved vertex array as well
  build_interleaved_vertices();
}

void sphere_drawer::build_interleaved_vertices() {
  std::vector<float>().swap(interleaved_vertices);

  std::size_t i, j;
  std::size_t count = vertices.size();
  for (i = 0, j = 0; i < count; i += 3, j += 2) {
    interleaved_vertices.push_back(vertices[i]);
    interleaved_vertices.push_back(vertices[i + 1]);
    interleaved_vertices.push_back(vertices[i + 2]);

    interleaved_vertices.push_back(normals[i]);
    interleaved_vertices.push_back(normals[i + 1]);
    interleaved_vertices.push_back(normals[i + 2]);

    interleaved_vertices.push_back(tex_coords[j]);
    interleaved_vertices.push_back(tex_coords[j + 1]);
  }
}

void sphere_drawer::add_vertex(float x, float y, float z) {
  vertices.push_back(x);
  vertices.push_back(y);
  vertices.push_back(z);
}

void sphere_drawer::add_normal(float nx, float ny, float nz) {
  normals.push_back(nx);
  normals.push_back(ny);
  normals.push_back(nz);
}

void sphere_drawer::add_tex_coord(float s, float t) {
  tex_coords.push_back(s);
  tex_coords.push_back(t);
}

void sphere_drawer::add_indices(unsigned int i1, unsigned int i2, unsigned int i3) {
  indices.push_back(i1);
  indices.push_back(i2);
  indices.push_back(i3);
}

std::vector<float> sphere_drawer::compute_face_normal(float x1, float y1, float z1,  // v1
                                                      float x2, float y2, float z2,  // v2
                                                      float x3, float y3, float z3)  // v3
{
  const float EPSILON = 0.000001f;

  std::vector<float> normal(3, 0.0f);  // default return value (0,0,0)
  float nx, ny, nz;

  // find 2 edge vectors: v1-v2, v1-v3
  float ex1 = x2 - x1;
  float ey1 = y2 - y1;
  float ez1 = z2 - z1;
  float ex2 = x3 - x1;
  float ey2 = y3 - y1;
  float ez2 = z3 - z1;

  // cross product: e1 x e2
  nx = ey1 * ez2 - ez1 * ey2;
  ny = ez1 * ex2 - ex1 * ez2;
  nz = ex1 * ey2 - ey1 * ex2;

  // normalize only if the length is > 0
  float length = sqrtf(nx * nx + ny * ny + nz * nz);
  if (length > EPSILON) {
    // normalize
    float lengthInv = 1.0f / length;
    normal[0] = nx * lengthInv;
    normal[1] = ny * lengthInv;
    normal[2] = nz * lengthInv;
  }

  return normal;
}
