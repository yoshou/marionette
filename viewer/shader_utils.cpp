#include "shader_utils.hpp"

#include <fstream>
#include <iostream>

namespace shader_utils {

int load_shader(GLuint shaderObj, const std::string& fileName) {
  std::ifstream ifs(fileName);
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
  GLint length = source.length();
  glShaderSource(shaderObj, 1, &sourcePtr, &length);

  return 0;
}

GLint load_program(const std::string& vertexFileName, const std::string& fragmentFileName) {
  GLuint vertShaderObj = glCreateShader(GL_VERTEX_SHADER);
  GLuint fragShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
  GLuint shader;
  GLint compiled, linked;

  if (load_shader(vertShaderObj, vertexFileName)) return -1;
  if (load_shader(fragShaderObj, fragmentFileName)) return -1;

  glCompileShader(vertShaderObj);
  glGetShaderiv(vertShaderObj, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    std::cout << "Vertex shader compile error" << std::endl;

    GLsizei bufSize;
    glGetShaderiv(vertShaderObj, GL_INFO_LOG_LENGTH, &bufSize);

    if (bufSize > 1) {
      GLchar* infoLog = (GLchar*)malloc(bufSize);
      GLsizei length;
      glGetShaderInfoLog(vertShaderObj, bufSize, &length, infoLog);
      std::cout << "InfoLog:" << std::endl << infoLog << std::endl << std::endl;
      free(infoLog);
    }

    return -1;
  }

  glCompileShader(fragShaderObj);
  glGetShaderiv(fragShaderObj, GL_COMPILE_STATUS, &compiled);
  if (!compiled) {
    std::cout << "Fragment shader compile error" << std::endl;

    GLsizei bufSize;
    glGetShaderiv(fragShaderObj, GL_INFO_LOG_LENGTH, &bufSize);

    if (bufSize > 1) {
      GLchar* infoLog = (GLchar*)malloc(bufSize);
      GLsizei length;
      glGetShaderInfoLog(fragShaderObj, bufSize, &length, infoLog);
      std::cout << "InfoLog:" << std::endl << infoLog << std::endl << std::endl;
      free(infoLog);
    }

    return -1;
  }

  shader = glCreateProgram();
  glAttachShader(shader, vertShaderObj);
  glAttachShader(shader, fragShaderObj);

  glLinkProgram(shader);
  glGetProgramiv(shader, GL_LINK_STATUS, &linked);
  if (!linked) {
    std::cout << "Shader link error" << std::endl;

    GLsizei bufSize;
    glGetProgramiv(shader, GL_INFO_LOG_LENGTH, &bufSize);

    if (bufSize > 1) {
      GLchar* infoLog = (GLchar*)malloc(bufSize);
      GLsizei length;
      glGetProgramInfoLog(shader, bufSize, &length, infoLog);
      std::cout << "InfoLog:" << std::endl << infoLog << std::endl << std::endl;
      free(infoLog);
    }

    return -1;
  }

  glDeleteShader(vertShaderObj);
  glDeleteShader(fragShaderObj);

  return shader;
}

}  // namespace shader_utils
