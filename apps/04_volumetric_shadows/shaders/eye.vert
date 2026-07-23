#version 450

// Eye-buffer pass: transform the slice with the eye matrix and also project it
// into the light buffer so the fragment shader can look up how much light
// reaches this slice.
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;

layout(location = 0) out vec3 vUVW;
layout(location = 1) out vec4 vLightClip;

layout(push_constant) uniform PushConstants {
    mat4 eyeMVP;
    mat4 lightMVP;
    float opacityCorrection;
} pc;

void main() {
    gl_Position = pc.eyeMVP * vec4(inPosition, 1.0);
    vUVW = inUVW;
    vLightClip = pc.lightMVP * vec4(inPosition, 1.0);
}
