#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;
layout(location = 0) out vec3 vUVW;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    float opacityCorrection; // reference / current sampling rate (Eq. 3)
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    vUVW = inUVW;
}
