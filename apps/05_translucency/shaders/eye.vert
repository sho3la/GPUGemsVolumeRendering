#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;
layout(location = 0) out vec3 vUVW;
layout(location = 1) out vec4 vLightClip;

layout(push_constant) uniform PushConstants {
    mat4 eyeMVP;
    mat4 lightMVP;
    vec4 params; // x = opacityCorrection, y = blurRadius
} pc;

void main() {
    gl_Position = pc.eyeMVP * vec4(inPosition, 1.0);
    vUVW = inUVW;
    vLightClip = pc.lightMVP * vec4(inPosition, 1.0);
}
