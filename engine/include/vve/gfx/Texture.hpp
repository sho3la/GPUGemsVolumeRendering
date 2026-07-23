#pragma once

#include "vve/gfx/VulkanCommon.hpp"

#include <vk_mem_alloc.h>

namespace vve::gfx {

class Context;

// A GPU image (1D/2D/3D) with an image view and an optional sampler. Used for
// volume data (3D), transfer functions (1D/2D), noise volumes and the offscreen
// light/eye buffers of the advanced techniques.
class Texture {
public:
    struct Desc {
        vk::Extent3D extent{1, 1, 1};
        vk::Format format = vk::Format::eR8Unorm;
        vk::ImageType type = vk::ImageType::e3D;
        vk::ImageUsageFlags usage = vk::ImageUsageFlagBits::eSampled |
                                    vk::ImageUsageFlagBits::eTransferDst;
        vk::Filter filter = vk::Filter::eLinear;
        vk::SamplerAddressMode addressMode =
            vk::SamplerAddressMode::eClampToEdge;
        bool sampler = true;
    };

    Texture() = default;
    Texture(Context& ctx, const Desc& desc);
    ~Texture();

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;
    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    // Uploads raw pixel data and transitions the image to shader-read-only.
    void uploadFromData(Context& ctx, const void* data, vk::DeviceSize byteSize);

    // Records a layout transition on an existing command buffer.
    void transition(vk::raii::CommandBuffer& cmd, vk::ImageLayout newLayout,
                    vk::PipelineStageFlags2 srcStage,
                    vk::PipelineStageFlags2 dstStage,
                    vk::AccessFlags2 srcAccess, vk::AccessFlags2 dstAccess);

    [[nodiscard]] vk::Image image() const { return m_image; }
    [[nodiscard]] vk::ImageView view() const { return *m_view; }
    [[nodiscard]] vk::Sampler sampler() const { return *m_sampler; }
    [[nodiscard]] vk::Format format() const { return m_desc.format; }
    [[nodiscard]] vk::Extent3D extent() const { return m_desc.extent; }
    [[nodiscard]] vk::ImageLayout layout() const { return m_layout; }

    [[nodiscard]] vk::DescriptorImageInfo descriptor() const {
        return vk::DescriptorImageInfo{*m_sampler, *m_view, m_layout};
    }

private:
    void destroy();

    Desc m_desc;
    VmaAllocator m_allocator = nullptr;
    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_allocation = nullptr;
    vk::raii::ImageView m_view{nullptr};
    vk::raii::Sampler m_sampler{nullptr};
    vk::ImageLayout m_layout = vk::ImageLayout::eUndefined;
};

} // namespace vve::gfx
