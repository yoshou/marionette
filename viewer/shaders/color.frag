#version 460 core

out vec4 FragColor;
in vec4 pixel_color;

void main(void)
{
    FragColor = pixel_color;
}