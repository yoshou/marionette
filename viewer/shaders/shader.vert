#version 450 core

layout (location = 0) in vec3 position;
layout (location = 1) in vec3 normal;
layout (location = 2) in vec2 texcoord;
layout (location = 3) in uvec4 joints;
layout (location = 4) in vec4 weights;

out vec2 v_texcoord;

uniform mat4 pvw;
uniform mat4 transforms[160];

void main(void)
{
    vec4 p = vec4(position, 1.0);
    vec4 blend_p = vec4(0, 0, 0, 0);
    for(int i = 0; i < 4; ++i)
    {
        mat4 transform = transforms[joints[i]];
        blend_p += weights[i] * transform * p;
    }
    gl_Position = pvw * vec4(blend_p.xyz, 1.0);
    v_texcoord = texcoord;
}