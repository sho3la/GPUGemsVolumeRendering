#include "vve/volume/SliceGeometryBuffers.hpp"

#include "vve/gfx/Context.hpp"

#include <algorithm>
#include <cstring>

namespace vve::volume {

SliceGeometryBuffers::SliceGeometryBuffers(gfx::Context& ctx,
                                           uint32_t framesInFlight,
                                           size_t maxVertices)
    : m_maxVertices(maxVertices) {
    const vk::DeviceSize capacity =
        sizeof(SliceProxyGeometry::Vertex) * maxVertices;
    m_buffers.reserve(framesInFlight);
    for (uint32_t i = 0; i < framesInFlight; ++i) {
        m_buffers.emplace_back(
            ctx, capacity, vk::BufferUsageFlagBits::eVertexBuffer,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                VMA_ALLOCATION_CREATE_MAPPED_BIT);
    }
}

uint32_t SliceGeometryBuffers::upload(
    uint32_t frameIndex,
    const std::vector<SliceProxyGeometry::Vertex>& verts) {
    size_t count = std::min(verts.size(), m_maxVertices);
    if (count == 0) return 0;
    std::memcpy(m_buffers[frameIndex].mapped(), verts.data(),
                count * sizeof(SliceProxyGeometry::Vertex));
    return static_cast<uint32_t>(count);
}

std::vector<vk::VertexInputBindingDescription>
SliceGeometryBuffers::bindings() {
    return {{0, sizeof(SliceProxyGeometry::Vertex),
             vk::VertexInputRate::eVertex}};
}

std::vector<vk::VertexInputAttributeDescription>
SliceGeometryBuffers::attributes() {
    return {{0, 0, vk::Format::eR32G32B32Sfloat,
             offsetof(SliceProxyGeometry::Vertex, position)},
            {1, 0, vk::Format::eR32G32B32Sfloat,
             offsetof(SliceProxyGeometry::Vertex, uvw)}};
}

} // namespace vve::volume
