#pragma once

#include <cstdint>
#include <vector>

namespace vve::volume {

class VolumeData;

// Precomputes per-voxel gradients via central differences (GPU Gems §39.4.1,
// Equation 2). The gradient doubles as the surface normal for illumination and
// as an axis for 2D transfer functions.
namespace GradientComputer {

// RGBA8: rgb = normalised gradient encoded to [0,1], a = density.
// Ready to upload as a single VK_FORMAT_R8G8B8A8_UNORM 3D texture so a shader
// gets value + normal in one trilinear fetch (the packing advised in §39.4.1).
std::vector<uint8_t> gradientAndDensityRGBA8(const VolumeData& volume);

// RG8: r = density, g = gradient magnitude. Feeds a 2D transfer function
// indexed by (value, |gradient|) as described in §39.4.3.
std::vector<uint8_t> valueAndGradientMagnitudeRG8(const VolumeData& volume);

} // namespace GradientComputer

} // namespace vve::volume
