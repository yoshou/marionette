#include "axis_drawer.hpp"
#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <array>

axis_drawer::axis_drawer()
{
    glm::vec3 max(1.f, 1.f, 1.f);
    {
        glm::vec3 v0(0.0f, 0.0f, 0.0f);
        glm::vec3 v1(max.x, 0.0f, 0.0f);
        glm::vec4 color(1, 0, 0, 1);
        add_color(color.r, color.g, color.b, color.a);
        add_vertex(v0.x, v0.y, v0.z);
        add_color(color.r, color.g, color.b, color.a);
        add_vertex(v1.x, v1.y, v1.z);
    }
    {
        glm::vec3 v0(0.0f, 0.0f, 0.0f);
        glm::vec3 v1(0.0f, max.y, 0.0f);
        glm::vec4 color(0, 1, 0, 1);
        add_color(color.r, color.g, color.b, color.a);
        add_vertex(v0.x, v0.y, v0.z);
        add_color(color.r, color.g, color.b, color.a);
        add_vertex(v1.x, v1.y, v1.z);
    }
    {
        glm::vec3 v0(0.0f, 0.0f, 0.0f);
        glm::vec3 v1(0.0f, 0.0f, max.z);
        glm::vec4 color(0, 0, 1, 1);
        add_color(color.r, color.g, color.b, color.a);
        add_vertex(v0.x, v0.y, v0.z);
        add_color(color.r, color.g, color.b, color.a);
        add_vertex(v1.x, v1.y, v1.z);
    }
}

void axis_drawer::add_vertex(float x, float y, float z)
{
    vertices.push_back(x);
    vertices.push_back(y);
    vertices.push_back(z);

    indices.push_back(indices.size());
    indices.push_back(indices.size());
    indices.push_back(indices.size());
}
void axis_drawer::add_color(float r, float g, float b, float a)
{
    colors.push_back(r);
    colors.push_back(g);
    colors.push_back(b);
    colors.push_back(a);
}

void axis_drawer::draw(glm::mat4 wvp) const
{
    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "pvw"), 1, GL_FALSE, &wvp[0][0]);

    glBindVertexArray(vao);
    glDrawElements(GL_LINES, (unsigned int)indices.size(), GL_UNSIGNED_INT, 0);
    glBindVertexArray(0);

    glUseProgram(0);
}

#include <fstream>
#include <string>
#include <iostream>

static int load_shader(GLuint shaderObj, std::string fileName)
{
    std::ifstream ifs(fileName);
    if (!ifs)
    {
        std::cout << "error" << std::endl;
        return -1;
    }

    std::string source;
    std::string line;
    while (getline(ifs, line))
    {
        source += line + "\n";
    }

    const GLchar *sourcePtr = (const GLchar *)source.c_str();
    GLint length = source.length();
    glShaderSource(shaderObj, 1, &sourcePtr, &length);

    return 0;
}

static GLint load_program(std::string vertexFileName, std::string fragmentFileName)
{
    // シェーダーオブジェクト作成
    GLuint vertShaderObj = glCreateShader(GL_VERTEX_SHADER);
    GLuint fragShaderObj = glCreateShader(GL_FRAGMENT_SHADER);
    GLuint shader;

    // シェーダーコンパイルとリンクの結果用変数
    GLint compiled, linked;

    /* シェーダーのソースプログラムの読み込み */
    if (load_shader(vertShaderObj, vertexFileName))
        return -1;
    if (load_shader(fragShaderObj, fragmentFileName))
        return -1;

    /* バーテックスシェーダーのソースプログラムのコンパイル */
    glCompileShader(vertShaderObj);
    glGetShaderiv(vertShaderObj, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE)
    {
        fprintf(stderr, "Compile error in vertex shader.\n");
        return -1;
    }

    /* フラグメントシェーダーのソースプログラムのコンパイル */
    glCompileShader(fragShaderObj);
    glGetShaderiv(fragShaderObj, GL_COMPILE_STATUS, &compiled);
    if (compiled == GL_FALSE)
    {
        fprintf(stderr, "Compile error in fragment shader.\n");
        return -1;
    }

    /* プログラムオブジェクトの作成 */
    shader = glCreateProgram();

    /* シェーダーオブジェクトのシェーダープログラムへの登録 */
    glAttachShader(shader, vertShaderObj);
    glAttachShader(shader, fragShaderObj);

    /* シェーダーオブジェクトの削除 */
    glDeleteShader(vertShaderObj);
    glDeleteShader(fragShaderObj);

    /* シェーダープログラムのリンク */
    glLinkProgram(shader);
    glGetProgramiv(shader, GL_LINK_STATUS, &linked);
    if (linked == GL_FALSE)
    {
        fprintf(stderr, "Link error.\n");
        return -1;
    }

    return shader;
}

void axis_drawer::initialize()
{
    shader = load_program("../viewer/shaders/color.vert", "../viewer/shaders/color.frag");

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), colors.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);

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
