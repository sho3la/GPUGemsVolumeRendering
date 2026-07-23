#version 450
layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inUVW;
layout(location = 0) out vec3 vUVW;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 params;  // x=time, y=noiseScale, z=coordPerturb, w=propPerturb
    vec4 params2; // x=opacityCorrection, y=flowSpeed
} pc;

void main() {
    gl_Position = pc.mvp * vec4(inPosition, 1.0);
    vUVW = inUVW;
}
