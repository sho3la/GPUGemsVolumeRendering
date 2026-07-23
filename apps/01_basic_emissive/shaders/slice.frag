#version 450

// Emissive-only direct volume rendering (GPU Gems Listing 39-1): each sample's
// emitted color is its density times a constant emissive color. Fragments are
// composited back-to-front with the premultiplied "over" operator (§39.4.3).

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 emissive;
    float densityScale;
} pc;

void main() {
    float a = clamp(texture(uVolume, vUVW).r * pc.densityScale, 0.0, 1.0);
    // Premultiplied output: rgb already scaled by alpha for correct blending.
    outColor = vec4(pc.emissive.rgb * a, a);
}
