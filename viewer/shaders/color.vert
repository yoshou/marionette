#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec4 color;

uniform mat4 pvw;
out vec4 pixel_color;

void main(void)
{
    gl_Position = pvw * vec4(position, 1.0);
    pixel_color = color;
}