#version 450

// Light-buffer pass of Algorithm 39-3. Only opacity matters here: with the
// premultiplied "over" operator and a zero output color, each slice multiplies
// the stored light color by (1 - alpha), attenuating the light that reaches
// deeper slices. The light buffer therefore holds the light still available
// after passing through all slices closer to the source.

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;
layout(set = 0, binding = 1) uniform sampler1D uTransfer;

layout(push_constant) uniform PushConstants {
    mat4 lightMVP;
    float opacityCorrection;
    float densityMin;       // density window [min,max]; keeps light in sync
    float densityMax;
} pc;

// 1 inside [min,max], fading to 0 just outside; never cuts the top at max>=1.
float densityGate(float value, float lo, float hi) {
    return smoothstep(lo, lo + 0.02, value) * (1.0 - smoothstep(hi, hi + 0.02, value));
}

void main() {
    float value = texture(uVolume, vUVW).r;
    float alpha = texture(uTransfer, value).a;
    // Keep only material inside the density window so removed skin/foliage no
    // longer blocks light; the isolated core then lights and shadows correctly.
    alpha *= densityGate(value, pc.densityMin, pc.densityMax);
    alpha = 1.0 - pow(max(1.0 - alpha, 0.0), pc.opacityCorrection);
    outColor = vec4(0.0, 0.0, 0.0, alpha); // color 0 => "over" attenuates target
}
