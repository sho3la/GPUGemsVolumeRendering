#version 450

// 1D post-classification: the scalar value indexes an RGBA transfer-function
// texture (GPU Gems §39.4.3). Opacity correction (Equation 3) keeps image
// intensity constant as the slice count changes.

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume; // r = value, g = |grad|
layout(set = 0, binding = 1) uniform sampler1D uTransfer;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    float opacityCorrection;
} pc;

void main() {
    float value = texture(uVolume, vUVW).r;
    vec4 tf = texture(uTransfer, value); // straight color + opacity

    // Opacity correction, then premultiply for the "over" operator.
    float a = 1.0 - pow(max(1.0 - tf.a, 0.0), pc.opacityCorrection);
    outColor = vec4(tf.rgb * a, a);
}
