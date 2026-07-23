#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include <vulkan/vulkan.h>

struct GLFWwindow;

namespace vve::core {

// Thin RAII wrapper over a GLFW window configured for Vulkan. Owns nothing
// Vulkan-specific; the surface is created on demand from a VkInstance so the
// windowing layer stays independent of the graphics device (SRP / DIP).
class Window {
public:
    struct Config {
        int width = 1280;
        int height = 720;
        std::string title = "VolumeEngine";
        bool resizable = true;
    };

    // Input callbacks a consumer (e.g. camera controller) can subscribe to.
    std::function<void(double x, double y)> onCursorMove;
    std::function<void(int button, int action, int mods)> onMouseButton;
    std::function<void(double xoffset, double yoffset)> onScroll;
    std::function<void(int key, int action, int mods)> onKey;
    std::function<void(int width, int height)> onResize;

    explicit Window(const Config& config);
    ~Window();

    Window(const Window&) = delete;
    Window& operator=(const Window&) = delete;

    [[nodiscard]] bool shouldClose() const;
    void pollEvents() const;

    // Required Vulkan instance extensions for surface creation (from GLFW).
    [[nodiscard]] std::vector<const char*> requiredInstanceExtensions() const;

    // Creates a VkSurfaceKHR for the given VkInstance. Returns the raw handle;
    // the caller wraps it (typically in a vk::raii::SurfaceKHR).
    [[nodiscard]] VkSurfaceKHR createSurface(VkInstance instance) const;

    [[nodiscard]] GLFWwindow* handle() const { return m_window; }
    [[nodiscard]] uint32_t width() const { return m_width; }
    [[nodiscard]] uint32_t height() const { return m_height; }
    [[nodiscard]] bool wasResized() const { return m_resized; }
    void clearResizedFlag() { m_resized = false; }

    // Blocks until the framebuffer has a non-zero size (used while minimized).
    void waitWhileMinimized() const;

private:
    GLFWwindow* m_window = nullptr;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_resized = false;

    static void framebufferResizeCallback(GLFWwindow*, int, int);
};

} // namespace vve::core
