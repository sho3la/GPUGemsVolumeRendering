#version 450

// Renders a slice from the light's point of view to accumulate shadowing.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;
layout(location = 0) out vec3 vUVW;

layout(push_constant) uniform PushConstants {
    mat4 lightMVP;
    float opacityCorrection;
} pc;

void main() {
    gl_Position = pc.lightMVP * vec4(inPosition, 1.0);
    vUVW = inUVW;
}
