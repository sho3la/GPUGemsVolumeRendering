#version 450

// 2D transfer function indexed by (value, gradient magnitude), the classic
// boundary-emphasising classification from GPU Gems §39.4.3. Material
// boundaries form arches in the value/gradient histogram; a 2D lookup can
// isolate them where a 1D function cannot.

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume; // r = value, g = |grad|
layout(set = 0, binding = 1) uniform sampler2D uTransfer2D;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    float opacityCorrection;
} pc;

void main() {
    vec2 vg = texture(uVolume, vUVW).rg; // (value, gradient magnitude)
    vec4 tf = texture(uTransfer2D, vg);

    float a = 1.0 - pow(max(1.0 - tf.a, 0.0), pc.opacityCorrection);
    outColor = vec4(tf.rgb * a, a);
}
