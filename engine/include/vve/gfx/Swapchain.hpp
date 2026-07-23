#pragma once

#include "vve/gfx/VulkanCommon.hpp"

#include <vector>

namespace vve::gfx {

class Context;

// Manages the swapchain and its per-image color views. Uses dynamic rendering,
// so it owns no render pass or framebuffers.
class Swapchain {
public:
    Swapchain(Context& ctx, uint32_t width, uint32_t height);

    // Rebuilds the swapchain for a new framebuffer size (on resize).
    void recreate(uint32_t width, uint32_t height);

    [[nodiscard]] vk::SwapchainKHR handle() const { return *m_swapchain; }

    // Acquires the next presentable image. May throw vk::OutOfDateKHRError.
    [[nodiscard]] std::pair<vk::Result, uint32_t> acquireNextImage(
        vk::Semaphore signalSemaphore, uint64_t timeout = UINT64_MAX) {
        auto rv = m_swapchain.acquireNextImage(timeout, signalSemaphore, nullptr);
        return {rv.result, rv.value};
    }

    [[nodiscard]] vk::Format format() const { return m_format; }
    [[nodiscard]] vk::Extent2D extent() const { return m_extent; }
    [[nodiscard]] uint32_t imageCount() const {
        return static_cast<uint32_t>(m_images.size());
    }
    [[nodiscard]] vk::Image image(uint32_t i) const { return m_images[i]; }
    [[nodiscard]] vk::ImageView view(uint32_t i) const { return *m_views[i]; }

private:
    void build(uint32_t width, uint32_t height);

    Context& m_ctx;
    vk::raii::SwapchainKHR m_swapchain{nullptr};
    std::vector<vk::Image> m_images;
    std::vector<vk::raii::ImageView> m_views;
    vk::Format m_format = vk::Format::eB8G8R8A8Unorm;
    vk::Extent2D m_extent{};
};

} // namespace vve::gfx
