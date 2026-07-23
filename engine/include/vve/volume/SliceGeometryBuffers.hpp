#pragma once

#include "vve/gfx/Buffer.hpp"
#include "vve/volume/SliceProxyGeometry.hpp"

#include <vector>

namespace vve::gfx { class Context; }

namespace vve::volume {

// Owns one persistently-mapped vertex buffer per frame-in-flight and streams
// freshly generated slice geometry into it each frame. Shared by every app so
// the double-buffering / upload logic lives in one place (SRP, DRY).
class SliceGeometryBuffers {
public:
    SliceGeometryBuffers(gfx::Context& ctx, uint32_t framesInFlight,
                         size_t maxVertices);

    // Copies `verts` into the buffer for `frameIndex` and returns the count.
    uint32_t upload(uint32_t frameIndex,
                    const std::vector<SliceProxyGeometry::Vertex>& verts);

    [[nodiscard]] vk::Buffer buffer(uint32_t frameIndex) const {
        return m_buffers[frameIndex].handle();
    }

    // Vertex input description matching SliceProxyGeometry::Vertex.
    static std::vector<vk::VertexInputBindingDescription> bindings();
    static std::vector<vk::VertexInputAttributeDescription> attributes();

private:
    std::vector<gfx::Buffer> m_buffers;
    size_t m_maxVertices;
};

} // namespace vve::volume
