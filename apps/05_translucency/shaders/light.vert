#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;
layout(location = 0) out vec3 vUVW;

layout(push_constant) uniform PushConstants {
    mat4 lightMVP;
    vec4 absorption;      // per-channel absorption color
    float opacityCorrection;
} pc;

void main() {
    gl_Position = pc.lightMVP * vec4(inPosition, 1.0);
    vUVW = inUVW;
}
