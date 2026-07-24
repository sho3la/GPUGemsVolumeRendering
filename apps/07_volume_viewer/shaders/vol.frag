#version 450

// Versatile volume viewer fragment shader. Supports two render modes selected by
// texelSize.w:
//   * DVR (0) - classified direct volume rendering with optional Blinn-Phong
//               gradient shading, composited with the premultiplied "over"
//               operator (the pipeline supplies back-to-front slice order).
//   * MIP (1) - maximum-intensity projection of the windowed scalar; the
//               pipeline uses a max blend, so slice order is irrelevant.
// A radiology-style window (window.xy) remaps the scalar before classification,
// and a density clip window (window.zw) hides values outside a range.

layout(location = 0) in vec3 vUVW;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;
layout(set = 0, binding = 1) uniform sampler1D uTransfer;

layout(push_constant) uniform PushConstants {
    mat4 mvp;
    vec4 texelSize;
    vec4 lightDir;
    vec4 viewDir;
    vec4 params;
    vec4 window;
} pc;

// 1 inside [lo,hi], fading to 0 just outside; never clips the top at hi >= 1.
float clipGate(float v, float lo, float hi) {
    return smoothstep(lo, lo + 0.02, v) * (1.0 - smoothstep(hi, hi + 0.02, v));
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
    float gate = clipGate(value, pc.window.z, pc.window.w);

    // Radiology window remaps the scalar onto the transfer-function domain.
    float lo = pc.window.x, hi = pc.window.y;
    float windowed = clamp((value - lo) / max(hi - lo, 1e-4), 0.0, 1.0);

    if (pc.texelSize.w > 0.5) {
        // MIP: brightest windowed sample wins (max blend).
        float m = windowed * gate;
        outColor = vec4(m, m, m, m);
        return;
    }

    // DVR.
    vec4 tf = texture(uTransfer, windowed);

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

    float surfaceness = smoothstep(0.02, 0.2, gradMag) * pc.params.z;
    vec3 lit = tf.rgb * intensity;
    vec3 color = mix(tf.rgb, lit, surfaceness);

    float a = 1.0 - pow(max(1.0 - tf.a, 0.0), pc.params.w);
    a *= gate;
    outColor = vec4(color * a, a);
}
