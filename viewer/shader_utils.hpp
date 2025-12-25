#pragma once

// clang-format off
#include <glad/glad.h>
#include <GLFW/glfw3.h>
// clang-format on

#include <string>

namespace shader_utils {

int load_shader(GLuint shaderObj, const std::string& fileName);

GLint load_program(const std::string& vertexFileName, const std::string& fragmentFileName);

}  // namespace shader_utils
