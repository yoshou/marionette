#include "drawer2d.hpp"

#include "grid_drawer.hpp"
#include <GL/glew.h>
#define GLFW_INCLUDE_GLU
#include <GLFW/glfw3.h>

drawer2d::drawer2d()
{
    {
        glm::vec3 v0(0.0f, 0.0f, -1.0f);
        glm::vec3 v1(1.0f, 0.0f, -1.0f);
        glm::vec3 v2(1.0f, 1.0f, -1.0f);
        glm::vec3 v3(0.0f, 1.0f, -1.0f);
        glm::vec4 color(1, 1, 1, 1);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v0.x, v0.y, v0.z);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v1.x, v1.y, v1.z);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v1.x, v1.y, v1.z);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v2.x, v2.y, v2.z);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v2.x, v2.y, v2.z);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v3.x, v3.y, v3.z);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v3.x, v3.y, v3.z);
        addColor(color.r, color.g, color.b, color.a);
        addVertex(v0.x, v0.y, v0.z);
    }
}

void drawer2d::addVertex(float x, float y, float z)
{
    vertices.push_back(x);
    vertices.push_back(y);
    vertices.push_back(z);

    indices.push_back(indices.size());
}
void drawer2d::addColor(float r, float g, float b, float a)
{
    colors.push_back(r);
    colors.push_back(g);
    colors.push_back(b);
    colors.push_back(a);
}

#include <glm/gtx/transform.hpp>

void drawer2d::draw_rect(const glm::vec2 &position, const glm::vec2 &size, const glm::vec4 &color) const
{
    glm::mat4 wvp = glm::translate(glm::vec3(-1.0f, 1.0f, 0.0f)) * glm::scale(glm::vec3(2.0f, -2.0f, 0.0f)) * glm::translate(glm::vec3(position.x, position.y, 0.0f)) * glm::scale(glm::vec3(size.x, size.y, 0.0f));

    glUseProgram(shader);
    glUniformMatrix4fv(glGetUniformLocation(shader, "pvw"), 1, GL_FALSE, &wvp[0][0]);

    // set line colour
    float values[] = {(float)color.x, (float)color.y, (float)color.z, (float)color.w};
    glUniform4fv(glGetUniformLocation(shader, "color"), 1, values);

    // set line colour
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, 0, 0);
    glEnableVertexAttribArray(1);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glDrawElements(GL_LINES, (unsigned int)indices.size(), GL_UNSIGNED_INT, 0);

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);

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

void drawer2d::initialize()
{
    shader = load_program("../viewer/shaders/position.vert", "../viewer/shaders/position.frag");

    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(float), vertices.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &color_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, color_buffer);
    glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(float), colors.data(), GL_STATIC_DRAW);

    glGenBuffers(1, &index_buffer);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, index_buffer);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), indices.data(), GL_STATIC_DRAW);
}
