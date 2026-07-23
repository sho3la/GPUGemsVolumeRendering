#pragma once

#include "vve/gfx/VulkanCommon.hpp"

#include <array>
#include <vector>

namespace vve::gfx {

class Context;
class Swapchain;

// Per-frame recording handle handed to the application.
struct Frame {
    vk::raii::CommandBuffer* cmd = nullptr;
    uint32_t imageIndex = 0;
    uint32_t frameIndex = 0;
    bool valid = false;
};

// Drives the acquire -> record -> submit -> present loop with N frames in
// flight. Rendering to the swapchain uses dynamic rendering; offscreen work can
// be recorded between beginFrame() and beginSwapchainPass().
class Renderer {
public:
    static constexpr uint32_t kFramesInFlight = 2;

    Renderer(Context& ctx, Swapchain& swapchain);

    // Acquires the next image. Returns an invalid Frame if the swapchain is out
    // of date (caller should recreate and skip this frame).
    Frame beginFrame();

    // Begins/ends dynamic rendering into the acquired swapchain image.
    void beginSwapchainPass(const Frame& frame,
                            const std::array<float, 4>& clearColor);
    void endSwapchainPass(const Frame& frame);

    // Ends recording, submits and presents.
    void endFrame(const Frame& frame);

    [[nodiscard]] bool needsRecreate() const { return m_needsRecreate; }
    void clearRecreateFlag() { m_needsRecreate = false; }
    void onSwapchainRecreated() { createSyncObjects(); }

private:
    void createCommandBuffers();
    void createSyncObjects();

    Context& m_ctx;
    Swapchain& m_swapchain;

    vk::raii::CommandPool m_commandPool{nullptr};
    std::vector<vk::raii::CommandBuffer> m_commandBuffers;

    std::vector<vk::raii::Semaphore> m_imageAvailable; // per frame-in-flight
    std::vector<vk::raii::Semaphore> m_renderFinished; // per swapchain image
    std::vector<vk::raii::Fence> m_inFlight;           // per frame-in-flight

    uint32_t m_currentFrame = 0;
    bool m_needsRecreate = false;
};

} // namespace vve::gfx
