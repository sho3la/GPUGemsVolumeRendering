#pragma once

#include "vve/core/Window.hpp"

#include <array>
#include <memory>
#include <string>

namespace vve::gfx {
class Context;
class Swapchain;
class Renderer;
struct Frame;
} // namespace vve::gfx

namespace vve::ui { class ImGuiLayer; }

namespace vve::core {

// Base application implementing the render-loop skeleton (template method).
// Chapter apps subclass this and override the on* hooks; the loop, swapchain
// recreation and ImGui plumbing are handled here (SRP + open/closed).
class Application {
public:
    struct Config {
        Window::Config window{};
        bool enableValidation = true;
    };

    explicit Application(const Config& config);
    virtual ~Application();

    void run();

protected:
    // Hooks for subclasses. Default implementations are no-ops.
    virtual void onInit() {}
    virtual void onUpdate(float /*dt*/) {}
    // Record work before the swapchain pass begins (offscreen render targets,
    // e.g. the light buffer of the half-angle shadow algorithm).
    virtual void onRecordOffscreen(gfx::Frame& /*frame*/) {}
    virtual void onRender(gfx::Frame& /*frame*/) {}
    virtual void onImGui() {}
    virtual void onShutdown() {}
    virtual void onResize(uint32_t /*w*/, uint32_t /*h*/) {}

    [[nodiscard]] Window& window() { return *m_window; }
    [[nodiscard]] gfx::Context& ctx() { return *m_context; }
    [[nodiscard]] gfx::Swapchain& swapchain() { return *m_swapchain; }
    [[nodiscard]] gfx::Renderer& renderer() { return *m_renderer; }

    std::array<float, 4> clearColor{0.02f, 0.02f, 0.03f, 1.0f};

private:
    void recreateSwapchain();

    std::unique_ptr<Window> m_window;
    std::unique_ptr<gfx::Context> m_context;
    std::unique_ptr<gfx::Swapchain> m_swapchain;
    std::unique_ptr<gfx::Renderer> m_renderer;
    std::unique_ptr<ui::ImGuiLayer> m_imgui;
};

} // namespace vve::core
