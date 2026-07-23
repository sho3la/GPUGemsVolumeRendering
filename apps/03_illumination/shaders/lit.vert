#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;
layout(location = 0) out vec3 vUVW;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 texelSize;  // xyz = 1/dim
    vec4 lightDir;   // xyz direction light->scene, w = ambient
    vec4 viewDir;    // xyz view direction,        w = shininess
    vec4 params;     // x=kd, y=ks, z=shadeStrength, w=opacityCorrection
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    vUVW = inUVW;
}
