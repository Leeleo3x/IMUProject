#version 430 core

uniform sampler2D tex_sampler;
in frag_texcoord;

void main(void){
    gl_color = texture2D(tex_sampler, frag_texcoord);
}