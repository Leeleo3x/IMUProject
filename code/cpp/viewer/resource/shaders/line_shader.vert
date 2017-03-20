attribute vec3 pos;
attribute vec4 v_color;

uniform mat4 m_mat;
uniform mat4 p_mat;
varying vec4 frag_color;

void main(void){
   gl_Position = p_mat * m_mat * vec4(pos, 1.0);
   frag_color = v_color;
}
