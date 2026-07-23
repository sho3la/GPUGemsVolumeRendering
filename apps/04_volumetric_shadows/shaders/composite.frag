#version 450

// Composites the accumulated eye buffer (premultiplied) over the cleared
// background using the "over" operator via the pipeline's blend state.
layout(location = 0) in vec2 vUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform sampler2D uEye;

void main() {
    outColor = texture(uEye, vUV);
}
