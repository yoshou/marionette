#version 460 core

layout (location = 0) in vec3 position;

uniform mat4 pvw;

void main(void)
{
    gl_Position = pvw * vec4(position, 1.0);
}