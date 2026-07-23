#version 450

// Procedural detail (GPU Gems §39.5.2). A small tiling noise volume adds
// high-frequency structure to a coarse macrostructure volume. Two techniques,
// both driven here:
//   * Coordinate perturbation - a noise vector displaces the volume lookup
//     coordinate, warping the field into cloud/smoke-like filaments (Fig. 39-14).
//   * Optical-property perturbation - noise modulates opacity after the
//     transfer function to break up smooth interpolation.
// Animation comes from scrolling the noise coordinate over time.

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;
layout(set = 0, binding = 1) uniform sampler1D uTransfer;
layout(set = 0, binding = 2) uniform sampler3D uNoise; // RGBA tiling noise

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 params;
    vec4 params2;
} pc;

vec3 fbmVector(vec3 p, float t) {
    // Two octaves of the noise volume, scrolled by time for animation.
    vec3 flow = vec3(0.13, 0.07, 0.05) * (t * pc.params2.y);
    vec3 n1 = texture(uNoise, p + flow).rgb;
    vec3 n2 = texture(uNoise, p * 2.03 - flow * 1.7).rgb;
    return (n1 - 0.5) + 0.5 * (n2 - 0.5);
}

void main() {
    float time = pc.params.x;
    float noiseScale = pc.params.y;
    float coordPerturb = pc.params.z;
    float propPerturb = pc.params.w;

    vec3 noise = fbmVector(vUVW * noiseScale, time);

    // (1) Coordinate perturbation.
    vec3 uvw = clamp(vUVW + noise * coordPerturb, 0.0, 1.0);
    float value = texture(uVolume, uvw).r;

    vec4 tf = texture(uTransfer, value);

    // (2) Optical-property perturbation: modulate opacity by a noise scalar.
    float noiseScalar = noise.x + noise.y + noise.z;
    float alpha = tf.a * (1.0 + propPerturb * noiseScalar);
    alpha = clamp(alpha, 0.0, 1.0);

    float a = 1.0 - pow(max(1.0 - alpha, 0.0), pc.params2.x);
    outColor = vec4(tf.rgb * a, a);
}
