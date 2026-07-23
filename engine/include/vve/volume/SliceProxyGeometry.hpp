#pragma once

#include <vector>

#include <glm/glm.hpp>

namespace vve::volume {

// Generates view-aligned (or arbitrary-axis) slice polygons that clip the
// volume's bounding box — the "proxy geometry" of GPU Gems §39.4.2,
// Algorithm 39-2. Working in object space avoids any matrix inversion: the
// caller supplies the slicing direction already expressed in object space, so
// the same routine serves both view-aligned slicing and the half-angle slicing
// used for volumetric shadows (§39.5.1).
namespace SliceProxyGeometry {

struct Vertex {
    glm::vec3 position; // object-space position (box in [boxMin, boxMax])
    glm::vec3 uvw;      // 3D texture coordinate in [0,1]
};

// A batch of slices with per-slice draw ranges, needed by techniques that must
// process one slice at a time (e.g. half-angle shadow slicing, §39.5.1).
struct SliceMesh {
    std::vector<Vertex> vertices;
    std::vector<uint32_t> firstVertex; // draw offset per slice
    std::vector<uint32_t> vertexCount; // draw count per slice
    [[nodiscard]] size_t sliceCount() const { return firstVertex.size(); }
};

// Same slicing as generate() but keeps slices individually addressable.
SliceMesh generateSliced(const glm::vec3& boxMin, const glm::vec3& boxMax,
                         const glm::vec3& sliceDir, int numSlices, bool reverse);

// Produces a triangle list. Slices are perpendicular to `sliceDir` (object
// space, need not be normalised) and are emitted ordered from the box extreme
// with the smallest projection onto `sliceDir` to the largest; set `reverse`
// to flip that ordering (e.g. for back-to-front vs front-to-back compositing).
std::vector<Vertex> generate(const glm::vec3& boxMin, const glm::vec3& boxMax,
                             const glm::vec3& sliceDir, int numSlices,
                             bool reverse);

} // namespace SliceProxyGeometry

} // namespace vve::volume
