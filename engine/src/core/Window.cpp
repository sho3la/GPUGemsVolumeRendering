#include "vve/core/Window.hpp"

#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#include <stdexcept>

namespace vve::core {

namespace {
int g_glfwRefCount = 0;

void ensureGlfwInitialized() {
    if (g_glfwRefCount++ == 0) {
        if (!glfwInit()) {
            throw std::runtime_error("Failed to initialize GLFW");
        }
    }
}

void releaseGlfw() {
    if (--g_glfwRefCount == 0) {
        glfwTerminate();
    }
}
} // namespace

Window::Window(const Config& config)
    : m_width(static_cast<uint32_t>(config.width))
    , m_height(static_cast<uint32_t>(config.height)) {
    ensureGlfwInitialized();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, config.resizable ? GLFW_TRUE : GLFW_FALSE);

    m_window = glfwCreateWindow(config.width, config.height,
                                config.title.c_str(), nullptr, nullptr);
    if (!m_window) {
        releaseGlfw();
        throw std::runtime_error("Failed to create GLFW window");
    }

    glfwSetWindowUserPointer(m_window, this);
    glfwSetFramebufferSizeCallback(m_window, framebufferResizeCallback);

    glfwSetCursorPosCallback(m_window, [](GLFWwindow* w, double x, double y) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        if (self->onCursorMove) self->onCursorMove(x, y);
    });
    glfwSetMouseButtonCallback(m_window, [](GLFWwindow* w, int b, int a, int m) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        if (self->onMouseButton) self->onMouseButton(b, a, m);
    });
    glfwSetScrollCallback(m_window, [](GLFWwindow* w, double x, double y) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        if (self->onScroll) self->onScroll(x, y);
    });
    glfwSetKeyCallback(m_window, [](GLFWwindow* w, int k, int, int a, int m) {
        auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
        if (self->onKey) self->onKey(k, a, m);
    });
}

Window::~Window() {
    if (m_window) {
        glfwDestroyWindow(m_window);
    }
    releaseGlfw();
}

void Window::framebufferResizeCallback(GLFWwindow* w, int width, int height) {
    auto* self = static_cast<Window*>(glfwGetWindowUserPointer(w));
    self->m_width = static_cast<uint32_t>(width);
    self->m_height = static_cast<uint32_t>(height);
    self->m_resized = true;
    if (self->onResize) self->onResize(width, height);
}

bool Window::shouldClose() const {
    return glfwWindowShouldClose(m_window);
}

void Window::pollEvents() const {
    glfwPollEvents();
}

std::vector<const char*> Window::requiredInstanceExtensions() const {
    uint32_t count = 0;
    const char** exts = glfwGetRequiredInstanceExtensions(&count);
    return {exts, exts + count};
}

VkSurfaceKHR Window::createSurface(VkInstance instance) const {
    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (glfwCreateWindowSurface(instance, m_window, nullptr, &surface) !=
        VK_SUCCESS) {
        throw std::runtime_error("Failed to create window surface");
    }
    return surface;
}

void Window::waitWhileMinimized() const {
    int w = 0, h = 0;
    glfwGetFramebufferSize(m_window, &w, &h);
    while (w == 0 || h == 0) {
        glfwGetFramebufferSize(m_window, &w, &h);
        glfwWaitEvents();
    }
}

} // namespace vve::core
