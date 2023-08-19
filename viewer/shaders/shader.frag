#version 450 core

out vec4 FragColor;
uniform sampler2D tex;
uniform vec4 diffuse;
in vec2 v_texcoord;

void main(void)
{
    vec4 color = texture(tex, v_texcoord);
    FragColor = color;
}