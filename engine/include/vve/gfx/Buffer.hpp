#pragma once

#include "vve/gfx/VulkanCommon.hpp"

#include <vk_mem_alloc.h>

namespace vve::gfx {

class Context;

// RAII wrapper over a VMA-allocated buffer. Move-only.
class Buffer {
public:
    Buffer() = default;
    Buffer(Context& ctx, vk::DeviceSize size, vk::BufferUsageFlags usage,
           VmaMemoryUsage memoryUsage,
           VmaAllocationCreateFlags flags = 0);
    ~Buffer();

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    // Copies `size` bytes into a host-visible/mapped buffer at `offset`.
    void upload(const void* data, vk::DeviceSize size, vk::DeviceSize offset = 0);

    // Creates a device-local buffer initialised from host data via a staging
    // buffer and an immediate copy submission.
    static Buffer createDeviceLocal(Context& ctx, const void* data,
                                    vk::DeviceSize size,
                                    vk::BufferUsageFlags usage);

    [[nodiscard]] vk::Buffer handle() const { return m_buffer; }
    [[nodiscard]] vk::DeviceSize size() const { return m_size; }
    [[nodiscard]] void* mapped() const { return m_mapped; }
    [[nodiscard]] bool valid() const { return m_buffer != VK_NULL_HANDLE; }

private:
    void destroy();

    VmaAllocator m_allocator = nullptr;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = nullptr;
    vk::DeviceSize m_size = 0;
    void* m_mapped = nullptr;
};

} // namespace vve::gfx
