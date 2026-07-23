#include "vve/gfx/Texture.hpp"

#include "vve/gfx/Buffer.hpp"
#include "vve/gfx/Context.hpp"

#include <utility>

namespace vve::gfx {

namespace {
vk::ImageViewType viewTypeFor(vk::ImageType type) {
    switch (type) {
        case vk::ImageType::e1D: return vk::ImageViewType::e1D;
        case vk::ImageType::e2D: return vk::ImageViewType::e2D;
        case vk::ImageType::e3D: return vk::ImageViewType::e3D;
        default: return vk::ImageViewType::e2D;
    }
}
} // namespace

Texture::Texture(Context& ctx, const Desc& desc)
    : m_desc(desc)
    , m_allocator(ctx.allocator()) {
    VkImageCreateInfo imageInfo{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    imageInfo.imageType = static_cast<VkImageType>(desc.type);
    imageInfo.format = static_cast<VkFormat>(desc.format);
    imageInfo.extent = desc.extent;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = static_cast<VkImageUsageFlags>(desc.usage);
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO;

    check(vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &m_image,
                         &m_allocation, nullptr),
          "vmaCreateImage");

    vk::ImageViewCreateInfo viewInfo{};
    viewInfo.image = m_image;
    viewInfo.viewType = viewTypeFor(desc.type);
    viewInfo.format = desc.format;
    viewInfo.subresourceRange = vk::ImageSubresourceRange{
        vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    m_view = vk::raii::ImageView{ctx.device(), viewInfo};

    if (desc.sampler) {
        vk::SamplerCreateInfo samplerInfo{};
        samplerInfo.magFilter = desc.filter;
        samplerInfo.minFilter = desc.filter;
        samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
        samplerInfo.addressModeU = desc.addressMode;
        samplerInfo.addressModeV = desc.addressMode;
        samplerInfo.addressModeW = desc.addressMode;
        samplerInfo.borderColor = vk::BorderColor::eFloatOpaqueBlack;
        m_sampler = vk::raii::Sampler{ctx.device(), samplerInfo};
    }
}

Texture::~Texture() { destroy(); }

void Texture::destroy() {
    m_view = nullptr;
    m_sampler = nullptr;
    if (m_image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
        m_image = VK_NULL_HANDLE;
        m_allocation = nullptr;
    }
}

Texture::Texture(Texture&& o) noexcept
    : m_desc(o.m_desc)
    , m_allocator(o.m_allocator)
    , m_image(o.m_image)
    , m_allocation(o.m_allocation)
    , m_view(std::move(o.m_view))
    , m_sampler(std::move(o.m_sampler))
    , m_layout(o.m_layout) {
    o.m_image = VK_NULL_HANDLE;
    o.m_allocation = nullptr;
}

Texture& Texture::operator=(Texture&& o) noexcept {
    if (this != &o) {
        destroy();
        m_desc = o.m_desc;
        m_allocator = o.m_allocator;
        m_image = o.m_image;
        m_allocation = o.m_allocation;
        m_view = std::move(o.m_view);
        m_sampler = std::move(o.m_sampler);
        m_layout = o.m_layout;
        o.m_image = VK_NULL_HANDLE;
        o.m_allocation = nullptr;
    }
    return *this;
}

void Texture::transition(vk::raii::CommandBuffer& cmd, vk::ImageLayout newLayout,
                         vk::PipelineStageFlags2 srcStage,
                         vk::PipelineStageFlags2 dstStage,
                         vk::AccessFlags2 srcAccess,
                         vk::AccessFlags2 dstAccess) {
    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = m_layout;
    barrier.newLayout = newLayout;
    barrier.image = m_image;
    barrier.subresourceRange =
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    vk::DependencyInfo dep{};
    dep.setImageMemoryBarriers(barrier);
    cmd.pipelineBarrier2(dep);
    m_layout = newLayout;
}

void Texture::uploadFromData(Context& ctx, const void* data,
                             vk::DeviceSize byteSize) {
    Buffer staging(ctx, byteSize, vk::BufferUsageFlagBits::eTransferSrc,
                   VMA_MEMORY_USAGE_AUTO,
                   VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                       VMA_ALLOCATION_CREATE_MAPPED_BIT);
    staging.upload(data, byteSize);

    ctx.submitImmediate([&](vk::raii::CommandBuffer& cmd) {
        transition(cmd, vk::ImageLayout::eTransferDstOptimal,
                   vk::PipelineStageFlagBits2::eTopOfPipe,
                   vk::PipelineStageFlagBits2::eCopy, {},
                   vk::AccessFlagBits2::eTransferWrite);

        vk::BufferImageCopy region{};
        region.imageSubresource = vk::ImageSubresourceLayers{
            vk::ImageAspectFlagBits::eColor, 0, 0, 1};
        region.imageExtent = m_desc.extent;
        cmd.copyBufferToImage(staging.handle(), m_image,
                              vk::ImageLayout::eTransferDstOptimal, region);

        transition(cmd, vk::ImageLayout::eShaderReadOnlyOptimal,
                   vk::PipelineStageFlagBits2::eCopy,
                   vk::PipelineStageFlagBits2::eFragmentShader,
                   vk::AccessFlagBits2::eTransferWrite,
                   vk::AccessFlagBits2::eShaderRead);
    });
}

} // namespace vve::gfx
