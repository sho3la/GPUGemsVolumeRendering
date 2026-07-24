#version 450

// Local volume illumination (GPU Gems §39.4.3, Equation 4). The gradient of the
// scalar field, computed on the fly by central differences (Equation 2), serves
// as the surface normal for a Blinn-Phong model. Shading is applied only where
// the gradient is strong (material boundaries), leaving homogeneous regions
// governed by the transfer function alone.

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;    // r = scalar value
layout(set = 0, binding = 1) uniform sampler1D uTransfer;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 texelSize;
    vec4 lightDir;
    vec4 viewDir;
    vec4 params;
    vec4 densityRange; // x=min, y=max
} pc;

// 1 inside [min,max], fading to 0 just outside; never cuts the top at max>=1.
float densityGate(float value, float lo, float hi) {
    return smoothstep(lo, lo + 0.02, value) * (1.0 - smoothstep(hi, hi + 0.02, value));
}

vec3 gradientCentralDiff(vec3 uvw) {
    vec3 ts = pc.texelSize.xyz;
    float dx = texture(uVolume, uvw + vec3(ts.x, 0, 0)).r -
               texture(uVolume, uvw - vec3(ts.x, 0, 0)).r;
    float dy = texture(uVolume, uvw + vec3(0, ts.y, 0)).r -
               texture(uVolume, uvw - vec3(0, ts.y, 0)).r;
    float dz = texture(uVolume, uvw + vec3(0, 0, ts.z)).r -
               texture(uVolume, uvw - vec3(0, 0, ts.z)).r;
    return 0.5 * vec3(dx, dy, dz);
}

void main() {
    float value = texture(uVolume, vUVW).r;
    vec4 tf = texture(uTransfer, value);

    vec3 grad = gradientCentralDiff(vUVW);
    float gradMag = length(grad);
    vec3 N = gradMag > 1e-5 ? -grad / gradMag : vec3(0.0, 0.0, 1.0);

    vec3 L = normalize(-pc.lightDir.xyz);
    vec3 V = normalize(-pc.viewDir.xyz);
    vec3 H = normalize(L + V);

    float ambient = pc.lightDir.w;
    float kd = pc.params.x;
    float ks = pc.params.y;
    float shininess = pc.viewDir.w;

    float diffuse = kd * max(dot(N, L), 0.0);
    float specular = ks * pow(max(dot(N, H), 0.0), shininess);
    float intensity = ambient + diffuse + specular;

    // Blend lit vs. unlit color by how "surface-like" the sample is.
    float surfaceness = smoothstep(0.02, 0.2, gradMag) * pc.params.z;
    vec3 litColor = tf.rgb * intensity;
    vec3 color = mix(tf.rgb, litColor, surfaceness);

    float a = 1.0 - pow(max(1.0 - tf.a, 0.0), pc.params.w);
    // Density window: keep only material inside [min,max].
    a *= densityGate(value, pc.densityRange.x, pc.densityRange.y);
    outColor = vec4(color * a, a);
}
