#include "shader_utils.hpp"

#include <fstream>
#include <iostream>

namespace shader_utils {

int load_shader(GLuint shader_obj, const std::string& file_name) {
  std::ifstream ifs(file_name);
  if (!ifs) {
    std::cout << "error" << std::endl;
    return -1;
  }

  std::string source;
  std::string line;
  while (getline(ifs, line)) {
    source += line + "\n";
  }

  const GLchar* sourcePtr = (const GLchar*)source.c_str();
  GLint length = static_cast<GLint>(source.length());
  glShaderSource(shader_obj, 1, &sourcePtr, &length);

  return 0;
}

GLint load_program(const std::string& vertex_file_name, const std::string& fragment_file_name) {
  GLuint vert_shader_obj = glCreateShader(GL_VERTEX_SHADER);
  GLuint frag_shader_obj = glCreateShader(GL_FRAGMENT_SHADER);
  GLuint shader;
  GLint compiled, linked;

  if (load_shader(vert_shader_obj, vertex_file_name)) return -1;
  if (load_shader(frag_shader_obj, fragment_file_name)) return -1;

  glCompileShader(vert_shader_obj);
  glGetShaderiv(vert_shader_obj, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    std::cout << "Vertex shader compile error" << std::endl;

    GLsizei buf_size;
    glGetShaderiv(vert_shader_obj, GL_INFO_LOG_LENGTH, &buf_size);

    if (buf_size > 1) {
      GLchar* info_log = (GLchar*)malloc(buf_size);
      GLsizei length;
      glGetShaderInfoLog(vert_shader_obj, buf_size, &length, info_log);
      std::cout << "InfoLog:" << std::endl << info_log << std::endl << std::endl;
      free(info_log);
    }

    return -1;
  }

  glCompileShader(frag_shader_obj);
  glGetShaderiv(frag_shader_obj, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    std::cout << "Fragment shader compile error" << std::endl;

    GLsizei buf_size;
    glGetShaderiv(frag_shader_obj, GL_INFO_LOG_LENGTH, &buf_size);

    if (buf_size > 1) {
      GLchar* info_log = (GLchar*)malloc(buf_size);
      GLsizei length;
      glGetShaderInfoLog(frag_shader_obj, buf_size, &length, info_log);
      std::cout << "InfoLog:" << std::endl << info_log << std::endl << std::endl;
      free(info_log);
    }

    return -1;
  }

  shader = glCreateProgram();
  glAttachShader(shader, vert_shader_obj);
  glAttachShader(shader, frag_shader_obj);

  glLinkProgram(shader);
  glGetProgramiv(shader, GL_LINK_STATUS, &linked);
  if (!linked) {
    std::cout << "Shader link error" << std::endl;

    GLsizei buf_size;
    glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &buf_size);

    if (buf_size > 1) {
      GLchar* info_log = (GLchar*)malloc(buf_size);
      GLsizei length;
      glGetProgramInfoLog(shader, buf_size, &length, info_log);
      std::cout << "InfoLog:" << std::endl << info_log << std::endl << std::endl;
      free(info_log);
    }

    return -1;
  }

  glDeleteShader(vert_shader_obj);
  glDeleteShader(frag_shader_obj);

  return shader;
}

}  // namespace shader_utils
