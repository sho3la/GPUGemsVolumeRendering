#pragma once

#include "vve/gfx/VulkanCommon.hpp"

#include <cstdint>
#include <functional>
#include <optional>

#include <vk_mem_alloc.h>

namespace vve::core { class Window; }

namespace vve::gfx {

struct QueueFamilyIndices {
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;
    [[nodiscard]] bool complete() const {
        return graphics.has_value() && present.has_value();
    }
};

// Owns the core Vulkan objects: instance, debug messenger, surface, physical
// and logical device, queues and the VMA allocator. Single responsibility:
// device/context lifetime. Rendering logic lives elsewhere (Renderer).
class Context {
public:
    struct Config {
        bool enableValidation = true;
        std::string appName = "VolumeEngine";
    };

    Context(core::Window& window, const Config& config);
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Accessors ---------------------------------------------------------------
    [[nodiscard]] vk::raii::Instance& instance() { return m_instance; }
    [[nodiscard]] vk::raii::PhysicalDevice& physicalDevice() { return m_physicalDevice; }
    [[nodiscard]] vk::raii::Device& device() { return m_device; }
    [[nodiscard]] vk::raii::SurfaceKHR& surface() { return m_surface; }
    [[nodiscard]] vk::raii::Queue& graphicsQueue() { return m_graphicsQueue; }
    [[nodiscard]] vk::raii::Queue& presentQueue() { return m_presentQueue; }
    [[nodiscard]] const QueueFamilyIndices& queueFamilies() const { return m_queues; }
    [[nodiscard]] VmaAllocator allocator() const { return m_allocator; }

    // Records and submits a one-shot command buffer, blocking until it
    // completes. Used for uploads and layout transitions during init.
    void submitImmediate(const std::function<void(vk::raii::CommandBuffer&)>& fn);

    void waitIdle() { m_device.waitIdle(); }

private:
    void createInstance(core::Window& window, const Config& config);
    void createSurface(core::Window& window);
    void pickPhysicalDevice();
    void createLogicalDevice(const Config& config);
    void createAllocator();

    vk::raii::Context m_context;
    vk::raii::Instance m_instance{nullptr};
    vk::raii::DebugUtilsMessengerEXT m_debugMessenger{nullptr};
    vk::raii::SurfaceKHR m_surface{nullptr};
    vk::raii::PhysicalDevice m_physicalDevice{nullptr};
    vk::raii::Device m_device{nullptr};
    vk::raii::Queue m_graphicsQueue{nullptr};
    vk::raii::Queue m_presentQueue{nullptr};

    // Immediate-submit resources.
    vk::raii::CommandPool m_uploadPool{nullptr};
    vk::raii::Fence m_uploadFence{nullptr};

    QueueFamilyIndices m_queues;
    VmaAllocator m_allocator{};
    bool m_validationEnabled = false;
};

} // namespace vve::gfx
