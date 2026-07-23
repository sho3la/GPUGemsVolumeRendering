#include "vve/gfx/Swapchain.hpp"

#include "vve/gfx/Context.hpp"

#include <algorithm>
#include <limits>

namespace vve::gfx {

namespace {
vk::SurfaceFormatKHR chooseFormat(
    const std::vector<vk::SurfaceFormatKHR>& formats) {
    for (const auto& f : formats) {
        if (f.format == vk::Format::eB8G8R8A8Unorm &&
            f.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear) {
            return f;
        }
    }
    return formats.front();
}

vk::PresentModeKHR choosePresentMode(
    const std::vector<vk::PresentModeKHR>& modes) {
    for (auto m : modes) {
        if (m == vk::PresentModeKHR::eMailbox) return m;
    }
    return vk::PresentModeKHR::eFifo; // always available
}
} // namespace

Swapchain::Swapchain(Context& ctx, uint32_t width, uint32_t height)
    : m_ctx(ctx) {
    build(width, height);
}

void Swapchain::recreate(uint32_t width, uint32_t height) {
    m_ctx.device().waitIdle();
    m_views.clear();
    m_images.clear();
    build(width, height);
}

void Swapchain::build(uint32_t width, uint32_t height) {
    auto& phys = m_ctx.physicalDevice();
    auto surface = *m_ctx.surface();

    auto caps = phys.getSurfaceCapabilitiesKHR(surface);
    auto formats = phys.getSurfaceFormatsKHR(surface);
    auto presentModes = phys.getSurfacePresentModesKHR(surface);

    auto surfaceFormat = chooseFormat(formats);
    m_format = surfaceFormat.format;

    if (caps.currentExtent.width != std::numeric_limits<uint32_t>::max()) {
        m_extent = caps.currentExtent;
    } else {
        m_extent.width = std::clamp(width, caps.minImageExtent.width,
                                    caps.maxImageExtent.width);
        m_extent.height = std::clamp(height, caps.minImageExtent.height,
                                     caps.maxImageExtent.height);
    }

    uint32_t desiredImages = caps.minImageCount + 1;
    if (caps.maxImageCount > 0 && desiredImages > caps.maxImageCount) {
        desiredImages = caps.maxImageCount;
    }

    vk::SwapchainCreateInfoKHR info{};
    info.surface = surface;
    info.minImageCount = desiredImages;
    info.imageFormat = surfaceFormat.format;
    info.imageColorSpace = surfaceFormat.colorSpace;
    info.imageExtent = m_extent;
    info.imageArrayLayers = 1;
    info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;

    const auto& q = m_ctx.queueFamilies();
    uint32_t indices[] = {*q.graphics, *q.present};
    if (q.graphics != q.present) {
        info.imageSharingMode = vk::SharingMode::eConcurrent;
        info.queueFamilyIndexCount = 2;
        info.pQueueFamilyIndices = indices;
    } else {
        info.imageSharingMode = vk::SharingMode::eExclusive;
    }

    info.preTransform = caps.currentTransform;
    info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    info.presentMode = choosePresentMode(presentModes);
    info.clipped = VK_TRUE;
    // Retire the previous swapchain instead of leaving it owning the surface,
    // otherwise recreation (e.g. on maximize) fails with ErrorNativeWindowInUse.
    info.oldSwapchain = *m_swapchain; // VK_NULL_HANDLE on the first build

    m_swapchain = vk::raii::SwapchainKHR{m_ctx.device(), info};
    m_images = m_swapchain.getImages();

    m_views.reserve(m_images.size());
    for (auto img : m_images) {
        vk::ImageViewCreateInfo viewInfo{};
        viewInfo.image = img;
        viewInfo.viewType = vk::ImageViewType::e2D;
        viewInfo.format = m_format;
        viewInfo.subresourceRange = vk::ImageSubresourceRange{
            vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
        m_views.emplace_back(m_ctx.device(), viewInfo);
    }
}

} // namespace vve::gfx
