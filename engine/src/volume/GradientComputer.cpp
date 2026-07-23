#include "vve/volume/GradientComputer.hpp"

#include "vve/volume/VolumeData.hpp"

#include <algorithm>
#include <cmath>

#include <glm/glm.hpp>

namespace vve::volume::GradientComputer {

namespace {
// Central difference with edge clamping (§39.4.1: repeat boundary voxels).
float sampleClamped(const VolumeData& v, int x, int y, int z) {
    x = std::clamp(x, 0, v.nx() - 1);
    y = std::clamp(y, 0, v.ny() - 1);
    z = std::clamp(z, 0, v.nz() - 1);
    return v.at(x, y, z);
}

glm::vec3 gradientAt(const VolumeData& v, int x, int y, int z) {
    // 1/(2h); voxel spacing folds into the later normalisation.
    return glm::vec3{
        (sampleClamped(v, x + 1, y, z) - sampleClamped(v, x - 1, y, z)) * 0.5f,
        (sampleClamped(v, x, y + 1, z) - sampleClamped(v, x, y - 1, z)) * 0.5f,
        (sampleClamped(v, x, y, z + 1) - sampleClamped(v, x, y, z - 1)) * 0.5f};
}
} // namespace

std::vector<uint8_t> gradientAndDensityRGBA8(const VolumeData& volume) {
    std::vector<uint8_t> out(volume.voxelCount() * 4);
    size_t i = 0;
    for (int z = 0; z < volume.nz(); ++z)
        for (int y = 0; y < volume.ny(); ++y)
            for (int x = 0; x < volume.nx(); ++x) {
                glm::vec3 g = gradientAt(volume, x, y, z);
                float len = glm::length(g);
                glm::vec3 n = len > 1e-6f ? g / len : glm::vec3(0.0f);
                // Encode signed normal [-1,1] -> [0,1].
                glm::vec3 enc = n * 0.5f + 0.5f;
                out[i++] = static_cast<uint8_t>(std::lround(enc.x * 255.0f));
                out[i++] = static_cast<uint8_t>(std::lround(enc.y * 255.0f));
                out[i++] = static_cast<uint8_t>(std::lround(enc.z * 255.0f));
                out[i++] = static_cast<uint8_t>(
                    std::lround(std::clamp(volume.at(x, y, z), 0.0f, 1.0f) * 255.0f));
            }
    return out;
}

std::vector<uint8_t> valueAndGradientMagnitudeRG8(const VolumeData& volume) {
    std::vector<uint8_t> out(volume.voxelCount() * 2);
    // Gradient magnitude of central differences maxes near sqrt(3)/2 for a
    // unit step; scale so typical boundaries span the [0,1] range.
    const float scale = 2.0f;
    size_t i = 0;
    for (int z = 0; z < volume.nz(); ++z)
        for (int y = 0; y < volume.ny(); ++y)
            for (int x = 0; x < volume.nx(); ++x) {
                float mag = glm::length(gradientAt(volume, x, y, z));
                out[i++] = static_cast<uint8_t>(
                    std::lround(std::clamp(volume.at(x, y, z), 0.0f, 1.0f) * 255.0f));
                out[i++] = static_cast<uint8_t>(
                    std::lround(std::clamp(mag * scale, 0.0f, 1.0f) * 255.0f));
            }
    return out;
}

} // namespace vve::volume::GradientComputer
