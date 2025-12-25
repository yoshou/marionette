#pragma once

#include <cmath>
#include <glm/glm.hpp>
#include <vector>

class sphere_drawer {
  static constexpr int MIN_SECTOR_COUNT = 3;
  static constexpr int MIN_STACK_COUNT = 2;
  int shader;

 public:
  sphere_drawer(int sector_count = 36, int stack_count = 18, bool smooth = true);
  ~sphere_drawer() {}

  void initialize();

  int get_sector_count() const { return sector_count; }
  int get_stack_count() const { return stack_count; }
  void set(int sector_count, int stack_count, bool smooth = true);
  void set_sector_count(int sector_count);
  void set_stack_count(int stack_count);
  void set_smooth(bool smooth);

  // for vertex data
  unsigned int get_vertex_count() const { return (unsigned int)vertices.size() / 3; }
  unsigned int get_normal_count() const { return (unsigned int)normals.size() / 3; }
  unsigned int get_tex_coord_count() const { return (unsigned int)tex_coords.size() / 2; }
  unsigned int get_index_count() const { return (unsigned int)indices.size(); }
  unsigned int get_line_index_count() const { return (unsigned int)line_indices.size(); }
  unsigned int get_triangle_count() const { return get_index_count() / 3; }
  unsigned int get_vertex_size() const { return (unsigned int)vertices.size() * sizeof(float); }
  unsigned int get_normal_size() const { return (unsigned int)normals.size() * sizeof(float); }
  unsigned int get_tex_coord_size() const {
    return (unsigned int)tex_coords.size() * sizeof(float);
  }
  unsigned int get_index_size() const {
    return (unsigned int)indices.size() * sizeof(unsigned int);
  }
  unsigned int get_line_index_size() const {
    return (unsigned int)line_indices.size() * sizeof(unsigned int);
  }
  const float *get_vertices() const { return vertices.data(); }
  const float *get_normals() const { return normals.data(); }
  const float *get_tex_coords() const { return tex_coords.data(); }
  const unsigned int *get_indices() const { return indices.data(); }
  const unsigned int *get_line_indices() const { return line_indices.data(); }

  // for interleaved vertices: V/N/T
  unsigned int get_interleaved_vertex_count() const { return get_vertex_count(); }  // # of vertices
  unsigned int get_interleaved_vertex_size() const {
    return (unsigned int)interleaved_vertices.size() * sizeof(float);
  }  // # of bytes
  int get_interleaved_stride() const { return interleaved_stride; }  // should be 32 bytes
  const float *get_interleaved_vertices() const { return interleaved_vertices.data(); }

  // draw in VertexArray mode
  void draw(glm::mat4 wvp, const float color[4]) const;                  // draw surface
  void draw_lines(glm::mat4 wvp, const float line_color[4]) const;       // draw lines only
  void draw_with_lines(glm::mat4 wvp, const float line_color[4]) const;  // draw surface and lines

 protected:
 private:
  // member functions
  void build_vertices_smooth();
  void build_vertices_flat();
  void build_interleaved_vertices();
  void clear_arrays();
  void add_vertex(float x, float y, float z);
  void add_normal(float x, float y, float z);
  void add_tex_coord(float s, float t);
  void add_indices(unsigned int i1, unsigned int i2, unsigned int i3);
  std::vector<float> compute_face_normal(float x1, float y1, float z1, float x2, float y2, float z2,
                                         float x3, float y3, float z3);

  // memeber vars
  int sector_count;  // longitude, # of slices
  int stack_count;   // latitude, # of stacks
  bool smooth;
  std::vector<float> vertices;
  std::vector<float> normals;
  std::vector<float> tex_coords;
  std::vector<unsigned int> indices;
  std::vector<unsigned int> line_indices;

  unsigned int vertex_buffer;
  unsigned int index_buffer;
  unsigned int line_index_buffer;

  unsigned int vao;
  unsigned int line_vao;

  // interleaved
  std::vector<float> interleaved_vertices;
  int interleaved_stride;  // # of bytes to hop to the next vertex (should be 32 bytes)
};
