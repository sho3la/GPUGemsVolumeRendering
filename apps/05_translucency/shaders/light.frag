#version 450

// Translucency light pass (GPU Gems §39.5.1). Instead of a single opacity, each
// slice attenuates the light per color channel. Multiplicative blending applies
// T = 1 - alpha * absorption to the stored light, so channels with lower
// absorption (e.g. red) penetrate deeper - the "colored translucency" /
// flashlight effect described in the chapter.

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;
layout(set = 0, binding = 1) uniform sampler1D uTransfer;

layout(push_constant) uniform PushConstants {
    mat4 lightMVP;
    vec4 absorption;
    float opacityCorrection;
} pc;

void main() {
    float value = texture(uVolume, vUVW).r;
    float alpha = texture(uTransfer, value).a;
    alpha = 1.0 - pow(max(1.0 - alpha, 0.0), pc.opacityCorrection);

    vec3 T = clamp(1.0 - alpha * pc.absorption.rgb, 0.0, 1.0);
    outColor = vec4(T, 1.0); // multiplicative blend attenuates the light buffer
}
