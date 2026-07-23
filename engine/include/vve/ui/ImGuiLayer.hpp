#pragma once

#include "vve/gfx/VulkanCommon.hpp"

namespace vve::core { class Window; }
namespace vve::gfx {
class Context;
class Swapchain;
}

namespace vve::ui {

// Wraps Dear ImGui's GLFW + Vulkan backends configured for dynamic rendering.
class ImGuiLayer {
public:
    ImGuiLayer(core::Window& window, gfx::Context& ctx, gfx::Swapchain& swapchain);
    ~ImGuiLayer();

    ImGuiLayer(const ImGuiLayer&) = delete;
    ImGuiLayer& operator=(const ImGuiLayer&) = delete;

    void beginFrame();                          // start a new ImGui frame
    void render(vk::raii::CommandBuffer& cmd);  // record draw data

private:
    gfx::Context& m_ctx;
    vk::raii::DescriptorPool m_descriptorPool{nullptr};
};

} // namespace vve::ui
