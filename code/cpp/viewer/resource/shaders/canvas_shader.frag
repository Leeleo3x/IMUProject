uniform sampler2D tex_sampler;
varying vec2 frag_texcoord;

void main(void){
    gl_FragColor = texture2D(tex_sampler, frag_texcoord);
}