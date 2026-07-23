#version 450

// Eye-buffer pass of Algorithm 39-3. The classified color is modulated by the
// light still available at this slice (sampled from the light buffer, which was
// attenuated by all closer slices) and composited into the eye buffer.

layout(location = 0) in vec3 vUVW;
layout(location = 1) in vec4 vLightClip;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler3D uVolume;
layout(set = 0, binding = 1) uniform sampler1D uTransfer;
layout(set = 0, binding = 2) uniform sampler2D uLightBuffer;

layout(push_constant) uniform PushConstants {
    mat4 eyeMVP;
    mat4 lightMVP;
    float opacityCorrection;
} pc;

void main() {
    float value = texture(uVolume, vUVW).r;
    vec4 tf = texture(uTransfer, value);

    // Project into the light buffer.
    vec3 ndc = vLightClip.xyz / vLightClip.w;
    vec2 luv = ndc.xy * 0.5 + 0.5;
    vec3 light = texture(uLightBuffer, luv).rgb;

    float a = 1.0 - pow(max(1.0 - tf.a, 0.0), pc.opacityCorrection);
    vec3 color = tf.rgb * light; // shadowed color
    outColor = vec4(color * a, a); // premultiplied
}
