#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <glm/glm.hpp>

namespace vve::volume {

// A CPU-side scalar field sampled on a regular grid. Values are normalised to
// [0,1]. This is the source data referenced throughout GPU Gems §39: it becomes
// a 3D texture, optionally augmented with a precomputed gradient (§39.4.1).
class VolumeData {
public:
    VolumeData() = default;
    VolumeData(int nx, int ny, int nz);

    [[nodiscard]] int nx() const { return m_nx; }
    [[nodiscard]] int ny() const { return m_ny; }
    [[nodiscard]] int nz() const { return m_nz; }
    [[nodiscard]] size_t voxelCount() const {
        return static_cast<size_t>(m_nx) * m_ny * m_nz;
    }

    float& at(int x, int y, int z) {
        return m_voxels[index(x, y, z)];
    }
    [[nodiscard]] float at(int x, int y, int z) const {
        return m_voxels[index(x, y, z)];
    }
    [[nodiscard]] const std::vector<float>& voxels() const { return m_voxels; }

    // World-space extent normalised so the longest axis is 1 (for proxy geometry).
    [[nodiscard]] glm::vec3 normalizedExtent() const;

    // Pack the field into an R8_UNORM byte array for texture upload.
    [[nodiscard]] std::vector<uint8_t> toR8() const;

    // --- Synthetic datasets (no external files needed) --------------------
    // A soft radial sphere/ball. `dim` is the resolution per axis.
    static VolumeData sphere(int dim);
    // The Marschner-Lobb signal, a standard reconstruction stress test.
    static VolumeData marschnerLobb(int dim);
    // A field of overlapping metaball blobs (cloud-like macrostructure).
    static VolumeData blobs(int dim, int count = 8, uint32_t seed = 1337);
    // A tangle/implicit surface: x^4-... classic CT-like isosurface demo.
    static VolumeData tangle(int dim);

    // Load an 8-bit raw volume (row-major x fastest). Values map to [0,1].
    static VolumeData loadRaw8(const std::string& path, int nx, int ny, int nz);

    // Load a raw volume of 8- or 16-bit unsigned voxels (little-endian, x
    // fastest) as used by the Open SciVis datasets. 8-bit maps by /255; 16-bit
    // is normalised by the maximum value present. If any dimension exceeds
    // `maxDim` the volume is subsampled to stay within GPU/upload limits.
    static VolumeData loadRaw(const std::string& path, int nx, int ny, int nz,
                              int bytesPerVoxel, int maxDim = 256);

private:
    [[nodiscard]] size_t index(int x, int y, int z) const {
        return static_cast<size_t>(z) * m_nx * m_ny +
               static_cast<size_t>(y) * m_nx + x;
    }

    int m_nx = 0, m_ny = 0, m_nz = 0;
    std::vector<float> m_voxels;
};

} // namespace vve::volume
