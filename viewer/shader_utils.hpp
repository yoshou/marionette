#pragma once

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <string>

namespace shader_utils {

int load_shader(GLuint shader_obj, const std::string& file_name);

GLint load_program(const std::string& vertex_file_name, const std::string& fragment_file_name);

}  // namespace shader_utils
