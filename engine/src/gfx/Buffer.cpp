#include "vve/gfx/Buffer.hpp"

#include "vve/gfx/Context.hpp"

#include <cstring>
#include <utility>

namespace vve::gfx {

Buffer::Buffer(Context& ctx, vk::DeviceSize size, vk::BufferUsageFlags usage,
               VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags flags)
    : m_allocator(ctx.allocator())
    , m_size(size) {
    VkBufferCreateInfo bufferInfo{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bufferInfo.size = size;
    bufferInfo.usage = static_cast<VkBufferUsageFlags>(usage);
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;
    allocInfo.flags = flags;

    VmaAllocationInfo info{};
    check(vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &m_buffer,
                          &m_allocation, &info),
          "vmaCreateBuffer");
    m_mapped = info.pMappedData; // non-null when MAPPED_BIT was requested
}

Buffer::~Buffer() { destroy(); }

void Buffer::destroy() {
    if (m_buffer != VK_NULL_HANDLE) {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        m_buffer = VK_NULL_HANDLE;
        m_allocation = nullptr;
        m_mapped = nullptr;
    }
}

Buffer::Buffer(Buffer&& o) noexcept
    : m_allocator(o.m_allocator)
    , m_buffer(o.m_buffer)
    , m_allocation(o.m_allocation)
    , m_size(o.m_size)
    , m_mapped(o.m_mapped) {
    o.m_buffer = VK_NULL_HANDLE;
    o.m_allocation = nullptr;
    o.m_mapped = nullptr;
}

Buffer& Buffer::operator=(Buffer&& o) noexcept {
    if (this != &o) {
        destroy();
        m_allocator = o.m_allocator;
        m_buffer = o.m_buffer;
        m_allocation = o.m_allocation;
        m_size = o.m_size;
        m_mapped = o.m_mapped;
        o.m_buffer = VK_NULL_HANDLE;
        o.m_allocation = nullptr;
        o.m_mapped = nullptr;
    }
    return *this;
}

void Buffer::upload(const void* data, vk::DeviceSize size, vk::DeviceSize offset) {
    if (m_mapped) {
        std::memcpy(static_cast<char*>(m_mapped) + offset, data, size);
        return;
    }
    // Persistent mapping was not requested: map transiently.
    void* dst = nullptr;
    check(vmaMapMemory(m_allocator, m_allocation, &dst), "vmaMapMemory");
    std::memcpy(static_cast<char*>(dst) + offset, data, size);
    vmaUnmapMemory(m_allocator, m_allocation);
}

Buffer Buffer::createDeviceLocal(Context& ctx, const void* data,
                                 vk::DeviceSize size,
                                 vk::BufferUsageFlags usage) {
    Buffer staging(ctx, size, vk::BufferUsageFlagBits::eTransferSrc,
                   VMA_MEMORY_USAGE_AUTO,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT);
    staging.upload(data, size);

    Buffer result(ctx, size, usage | vk::BufferUsageFlagBits::eTransferDst,
                  VMA_MEMORY_USAGE_AUTO);

    ctx.submitImmediate([&](vk::raii::CommandBuffer& cmd) {
        cmd.copyBuffer(staging.handle(), result.handle(),
                       vk::BufferCopy{0, 0, size});
    });
    return result;
}

} // namespace vve::gfx
