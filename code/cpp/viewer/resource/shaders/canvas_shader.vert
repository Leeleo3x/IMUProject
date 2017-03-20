//#version 330 core

attribute vec3 pos;
attribute vec2 texcoord;
uniform mat4 m_mat;
uniform mat4 p_mat;

varying vec2 frag_texcoord;

void main(void){
    gl_Position = p_mat * m_mat * vec4(pos, 1.0);
    frag_texcoord = texcoord;
}