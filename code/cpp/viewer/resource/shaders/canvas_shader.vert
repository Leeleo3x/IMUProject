#version 430 core

in vec3 pos;
in vec2 texcoord;
uniform in mat4 m_mat;
uniform in mat4 p_mat;

out vec2 frag_texcoord;

void main(void){
    gl_position = p_mat * m_mat * vec4(pos, 1.0);
    frag_texcoord = texcoord;
}