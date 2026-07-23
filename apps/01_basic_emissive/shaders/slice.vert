#version 450

// View-aligned proxy-geometry vertex (GPU Gems §39.4.2). Positions arrive in
// object space; the 3D texture coordinate travels with each vertex.

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;

layout(location = 0) out vec3 vUVW;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 emissive;     // rgb emission, a unused
    float densityScale;
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    vUVW = inUVW;
}
