#version 450

// Translucency eye pass. The incoming light is gathered from the light buffer
// with a small blur kernel, approximating the lateral spreading of forward-
// scattered light through the medium (GPU Gems §39.5.1, blur radius r = d*tan(phi),
// Equation 9). Blurring the light buffer - not the volume - decouples the light
// transport resolution from the data grid.

layout(location = 0) in vec3 vUVW;
layout(location = 1) in vec4 vLightClip;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;
layout(set = 0, binding = 1) uniform sampler1D uTransfer;
layout(set = 0, binding = 2) uniform sampler2D uLightBuffer;

layout(push_constant) uniform PushConstants {
    mat4 eyeMVP;
    mat4 lightMVP;
    vec4 params; // x = opacityCorrection, y = blurRadius
} pc;

vec3 sampleLightBlurred(vec2 uv, float radius) {
    // 3x3 tent-weighted taps.
    vec3 sum = vec3(0.0);
    float wsum = 0.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float w = (2.0 - abs(float(i))) * (2.0 - abs(float(j)));
            sum += w * texture(uLightBuffer, uv + vec2(i, j) * radius).rgb;
            wsum += w;
        }
    }
    return sum / wsum;
}

void main() {
    float value = texture(uVolume, vUVW).r;
    vec4 tf = texture(uTransfer, value);

    vec3 ndc = vLightClip.xyz / vLightClip.w;
    vec2 luv = ndc.xy * 0.5 + 0.5;
    vec3 light = sampleLightBlurred(luv, pc.params.y);

    float a = 1.0 - pow(max(1.0 - tf.a, 0.0), pc.params.x);
    vec3 color = tf.rgb * light;
    outColor = vec4(color * a, a);
}
