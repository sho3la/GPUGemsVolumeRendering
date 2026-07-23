#include "vve/volume/VolumeData.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <random>
#include <stdexcept>

namespace vve::volume {

VolumeData::VolumeData(int nx, int ny, int nz)
    : m_nx(nx), m_ny(ny), m_nz(nz), m_voxels(static_cast<size_t>(nx) * ny * nz, 0.0f) {}

glm::vec3 VolumeData::normalizedExtent() const {
    glm::vec3 dims{static_cast<float>(m_nx), static_cast<float>(m_ny),
                   static_cast<float>(m_nz)};
    float longest = std::max({dims.x, dims.y, dims.z});
    return dims / longest;
}

std::vector<uint8_t> VolumeData::toR8() const {
    std::vector<uint8_t> out(m_voxels.size());
    for (size_t i = 0; i < m_voxels.size(); ++i) {
        float v = std::clamp(m_voxels[i], 0.0f, 1.0f);
        out[i] = static_cast<uint8_t>(std::lround(v * 255.0f));
    }
    return out;
}

VolumeData VolumeData::sphere(int dim) {
    VolumeData vol(dim, dim, dim);
    const float r = 0.4f;
    for (int z = 0; z < dim; ++z)
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x) {
                glm::vec3 p{(x + 0.5f) / dim - 0.5f, (y + 0.5f) / dim - 0.5f,
                            (z + 0.5f) / dim - 0.5f};
                float d = glm::length(p);
                // Smooth falloff so classification has a boundary to work with.
                float v = std::clamp(1.0f - (d / r), 0.0f, 1.0f);
                vol.at(x, y, z) = v * v;
            }
    return vol;
}

VolumeData VolumeData::marschnerLobb(int dim) {
    VolumeData vol(dim, dim, dim);
    const float fM = 6.0f, alpha = 0.25f;
    auto rho = [&](float r) {
        return std::cos(2.0f * 3.14159265f * fM * std::cos(3.14159265f * r / 2.0f));
    };
    for (int z = 0; z < dim; ++z)
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x) {
                float px = 2.0f * (x + 0.5f) / dim - 1.0f;
                float py = 2.0f * (y + 0.5f) / dim - 1.0f;
                float pz = 2.0f * (z + 0.5f) / dim - 1.0f;
                float r = std::sqrt(px * px + py * py);
                float v = (1.0f - std::sin(3.14159265f * pz / 2.0f) +
                           alpha * (1.0f + rho(r))) /
                          (2.0f * (1.0f + alpha));
                vol.at(x, y, z) = std::clamp(v, 0.0f, 1.0f);
            }
    return vol;
}

VolumeData VolumeData::blobs(int dim, int count, uint32_t seed) {
    VolumeData vol(dim, dim, dim);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> pos(0.25f, 0.75f);
    std::uniform_real_distribution<float> rad(0.08f, 0.18f);

    std::vector<glm::vec4> spheres; // xyz center, w radius
    for (int i = 0; i < count; ++i)
        spheres.emplace_back(pos(rng), pos(rng), pos(rng), rad(rng));

    for (int z = 0; z < dim; ++z)
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x) {
                glm::vec3 p{(x + 0.5f) / dim, (y + 0.5f) / dim, (z + 0.5f) / dim};
                float field = 0.0f;
                for (const auto& s : spheres) {
                    float d2 = glm::dot(p - glm::vec3(s), p - glm::vec3(s));
                    float r2 = s.w * s.w;
                    field += r2 / (d2 + 1e-4f); // metaball potential
                }
                vol.at(x, y, z) = std::clamp(field * 0.15f, 0.0f, 1.0f);
            }
    return vol;
}

VolumeData VolumeData::tangle(int dim) {
    VolumeData vol(dim, dim, dim);
    for (int z = 0; z < dim; ++z)
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x) {
                float px = 3.0f * (2.0f * (x + 0.5f) / dim - 1.0f);
                float py = 3.0f * (2.0f * (y + 0.5f) / dim - 1.0f);
                float pz = 3.0f * (2.0f * (z + 0.5f) / dim - 1.0f);
                float f = px * px * px * px - 5.0f * px * px +
                          py * py * py * py - 5.0f * py * py +
                          pz * pz * pz * pz - 5.0f * pz * pz + 11.8f;
                // Map the implicit function to a soft density around the surface.
                float v = std::clamp(1.0f - std::abs(f) * 0.1f, 0.0f, 1.0f);
                vol.at(x, y, z) = v;
            }
    return vol;
}

VolumeData VolumeData::loadRaw(const std::string& path, int nx, int ny, int nz,
                               int bytesPerVoxel, int maxDim) {
    if (bytesPerVoxel != 1 && bytesPerVoxel != 2) {
        throw std::runtime_error("loadRaw: only 8- or 16-bit voxels supported");
    }
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open raw volume: " + path);

    size_t count = static_cast<size_t>(nx) * ny * nz;
    std::vector<uint8_t> raw(count * bytesPerVoxel);
    file.read(reinterpret_cast<char*>(raw.data()),
              static_cast<std::streamsize>(raw.size()));
    if (static_cast<size_t>(file.gcount()) != raw.size()) {
        throw std::runtime_error("Raw volume smaller than expected: " + path);
    }

    // Decode to normalised float.
    std::vector<float> values(count);
    if (bytesPerVoxel == 1) {
        for (size_t i = 0; i < count; ++i) values[i] = raw[i] / 255.0f;
    } else {
        uint16_t maxVal = 1;
        const auto* p16 = reinterpret_cast<const uint16_t*>(raw.data());
        for (size_t i = 0; i < count; ++i) maxVal = std::max(maxVal, p16[i]);
        float inv = 1.0f / static_cast<float>(maxVal);
        for (size_t i = 0; i < count; ++i) values[i] = p16[i] * inv;
    }

    // Optional subsampling so large scans fit comfortably in a 3D texture.
    int stride = 1;
    int longest = std::max({nx, ny, nz});
    while (longest / stride > maxDim) ++stride;

    int dx = (nx + stride - 1) / stride;
    int dy = (ny + stride - 1) / stride;
    int dz = (nz + stride - 1) / stride;
    VolumeData vol(dx, dy, dz);
    for (int z = 0; z < dz; ++z)
        for (int y = 0; y < dy; ++y)
            for (int x = 0; x < dx; ++x) {
                size_t src = static_cast<size_t>(z * stride) * nx * ny +
                             static_cast<size_t>(y * stride) * nx +
                             static_cast<size_t>(x * stride);
                vol.at(x, y, z) = values[src];
            }
    return vol;
}

VolumeData VolumeData::loadRaw8(const std::string& path, int nx, int ny, int nz) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open raw volume: " + path);

    size_t count = static_cast<size_t>(nx) * ny * nz;
    std::vector<uint8_t> raw(count);
    file.read(reinterpret_cast<char*>(raw.data()),
              static_cast<std::streamsize>(count));
    if (static_cast<size_t>(file.gcount()) != count) {
        throw std::runtime_error("Raw volume smaller than expected: " + path);
    }

    VolumeData vol(nx, ny, nz);
    for (size_t i = 0; i < count; ++i) {
        vol.m_voxels[i] = raw[i] / 255.0f;
    }
    return vol;
}

} // namespace vve::volume
