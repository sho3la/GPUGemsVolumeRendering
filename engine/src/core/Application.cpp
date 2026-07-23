#include "vve/core/Application.hpp"

#include "vve/core/Log.hpp"
#include "vve/gfx/Context.hpp"
#include "vve/gfx/Renderer.hpp"
#include "vve/gfx/Swapchain.hpp"
#include "vve/ui/ImGuiLayer.hpp"

#include <chrono>

namespace vve::core {

Application::Application(const Config& config) {
    m_window = std::make_unique<Window>(config.window);

    gfx::Context::Config ctxConfig{};
    ctxConfig.enableValidation = config.enableValidation;
    ctxConfig.appName = config.window.title;
    m_context = std::make_unique<gfx::Context>(*m_window, ctxConfig);

    m_swapchain = std::make_unique<gfx::Swapchain>(
        *m_context, m_window->width(), m_window->height());
    m_renderer = std::make_unique<gfx::Renderer>(*m_context, *m_swapchain);
    m_imgui = std::make_unique<ui::ImGuiLayer>(*m_window, *m_context, *m_swapchain);
}

Application::~Application() {
    if (m_context) m_context->waitIdle();
}

void Application::recreateSwapchain() {
    m_window->waitWhileMinimized();
    m_swapchain->recreate(m_window->width(), m_window->height());
    m_renderer->onSwapchainRecreated();
    m_renderer->clearRecreateFlag();
    m_window->clearResizedFlag();
    onResize(m_window->width(), m_window->height());
}

void Application::run() {
    onInit();

    using clock = std::chrono::high_resolution_clock;
    auto last = clock::now();

    while (!m_window->shouldClose()) {
        m_window->pollEvents();

        auto now = clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        last = now;

        if (m_window->wasResized() || m_renderer->needsRecreate()) {
            recreateSwapchain();
        }

        onUpdate(dt);

        gfx::Frame frame = m_renderer->beginFrame();
        if (!frame.valid) {
            recreateSwapchain();
            continue;
        }

        m_imgui->beginFrame();
        onImGui();

        onRecordOffscreen(frame);

        m_renderer->beginSwapchainPass(frame, clearColor);
        onRender(frame);
        m_imgui->render(*frame.cmd);
        m_renderer->endSwapchainPass(frame);

        m_renderer->endFrame(frame);
    }

    m_context->waitIdle();
    onShutdown();
}

} // namespace vve::core
