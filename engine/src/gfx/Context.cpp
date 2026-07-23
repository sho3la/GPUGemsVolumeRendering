#include "vve/gfx/Context.hpp"

#include "vve/core/Log.hpp"
#include "vve/core/Window.hpp"

#include <cstring>
#include <set>
#include <vector>

namespace vve::gfx {

namespace {

const std::vector<const char*> kValidationLayers = {
    "VK_LAYER_KHRONOS_validation"};

const std::vector<const char*> kDeviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME};

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT severity,
    VkDebugUtilsMessageTypeFlagsEXT,
    const VkDebugUtilsMessengerCallbackDataEXT* data, void*) {
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        vve::log::warn("[validation] %s", data->pMessage);
    }
    return VK_FALSE;
}

bool validationLayersSupported(const vk::raii::Context& ctx) {
    auto available = ctx.enumerateInstanceLayerProperties();
    for (const char* wanted : kValidationLayers) {
        bool found = false;
        for (const auto& prop : available) {
            if (std::strcmp(wanted, prop.layerName) == 0) { found = true; break; }
        }
        if (!found) return false;
    }
    return true;
}

vk::DebugUtilsMessengerCreateInfoEXT makeDebugCreateInfo() {
    using Sev = vk::DebugUtilsMessageSeverityFlagBitsEXT;
    using Type = vk::DebugUtilsMessageTypeFlagBitsEXT;
    return vk::DebugUtilsMessengerCreateInfoEXT{
        {},
        Sev::eWarning | Sev::eError,
        Type::eGeneral | Type::eValidation | Type::ePerformance,
        debugCallback};
}

} // namespace

Context::Context(core::Window& window, const Config& config) {
    createInstance(window, config);
    createSurface(window);
    pickPhysicalDevice();
    createLogicalDevice(config);
    createAllocator();

    // Immediate-submit command pool + fence.
    m_uploadPool = vk::raii::CommandPool{
        m_device,
        vk::CommandPoolCreateInfo{vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
                                  *m_queues.graphics}};
    m_uploadFence = vk::raii::Fence{m_device, vk::FenceCreateInfo{}};
}

Context::~Context() {
    if (m_allocator) {
        vmaDestroyAllocator(m_allocator);
        m_allocator = nullptr;
    }
    // Remaining vk::raii members destroy themselves in reverse declaration order.
}

void Context::createInstance(core::Window& window, const Config& config) {
    m_validationEnabled =
        config.enableValidation && validationLayersSupported(m_context);

    vk::ApplicationInfo appInfo{config.appName.c_str(), VK_MAKE_VERSION(1, 0, 0),
                                "VolumeEngine", VK_MAKE_VERSION(1, 0, 0),
                                VK_API_VERSION_1_3};

    auto extensions = window.requiredInstanceExtensions();
    if (m_validationEnabled) {
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    vk::InstanceCreateInfo info{{}, &appInfo};
    info.setPEnabledExtensionNames(extensions);

    auto debugInfo = makeDebugCreateInfo();
    if (m_validationEnabled) {
        info.setPEnabledLayerNames(kValidationLayers);
        info.pNext = &debugInfo;
    }

    m_instance = vk::raii::Instance{m_context, info};

    if (m_validationEnabled) {
        m_debugMessenger =
            vk::raii::DebugUtilsMessengerEXT{m_instance, debugInfo};
        log::info("Vulkan validation layers enabled");
    }
}

void Context::createSurface(core::Window& window) {
    VkSurfaceKHR raw = window.createSurface(*m_instance);
    m_surface = vk::raii::SurfaceKHR{m_instance, raw};
}

void Context::pickPhysicalDevice() {
    vk::raii::PhysicalDevices devices{m_instance};
    if (devices.empty()) {
        throw std::runtime_error("No Vulkan-capable GPU found");
    }

    auto findQueues = [&](const vk::raii::PhysicalDevice& dev) {
        QueueFamilyIndices indices;
        auto props = dev.getQueueFamilyProperties();
        for (uint32_t i = 0; i < props.size(); ++i) {
            if (props[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                indices.graphics = i;
            }
            if (dev.getSurfaceSupportKHR(i, *m_surface)) {
                indices.present = i;
            }
            if (indices.complete()) break;
        }
        return indices;
    };

    // Prefer discrete GPUs; fall back to the first device that fits.
    vk::raii::PhysicalDevice* fallback = nullptr;
    for (auto& dev : devices) {
        auto indices = findQueues(dev);
        if (!indices.complete()) continue;

        m_queues = indices;
        if (dev.getProperties().deviceType ==
            vk::PhysicalDeviceType::eDiscreteGpu) {
            m_physicalDevice = dev;
            log::info("Using GPU: %s", dev.getProperties().deviceName.data());
            return;
        }
        if (!fallback) fallback = &dev;
    }

    if (!fallback) {
        throw std::runtime_error("No suitable GPU with graphics + present queue");
    }
    m_queues = findQueues(*fallback);
    m_physicalDevice = *fallback;
    log::info("Using GPU: %s", fallback->getProperties().deviceName.data());
}

void Context::createLogicalDevice(const Config&) {
    std::set<uint32_t> uniqueFamilies = {*m_queues.graphics, *m_queues.present};
    float priority = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueInfos;
    for (uint32_t family : uniqueFamilies) {
        queueInfos.push_back(vk::DeviceQueueCreateInfo{{}, family, 1, &priority});
    }

    vk::PhysicalDeviceFeatures features{};
    features.samplerAnisotropy = VK_TRUE;
    features.fillModeNonSolid = VK_TRUE;

    vk::PhysicalDeviceVulkan13Features features13{};
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;

    vk::DeviceCreateInfo info{};
    info.setQueueCreateInfos(queueInfos);
    info.setPEnabledExtensionNames(kDeviceExtensions);
    info.setPEnabledFeatures(&features);
    info.pNext = &features13;
    if (m_validationEnabled) {
        info.setPEnabledLayerNames(kValidationLayers);
    }

    m_device = vk::raii::Device{m_physicalDevice, info};
    m_graphicsQueue = m_device.getQueue(*m_queues.graphics, 0);
    m_presentQueue = m_device.getQueue(*m_queues.present, 0);
}

void Context::createAllocator() {
    VmaAllocatorCreateInfo info{};
    info.physicalDevice = *m_physicalDevice;
    info.device = *m_device;
    info.instance = *m_instance;
    info.vulkanApiVersion = VK_API_VERSION_1_3;

    if (vmaCreateAllocator(&info, &m_allocator) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create VMA allocator");
    }
}

void Context::submitImmediate(
    const std::function<void(vk::raii::CommandBuffer&)>& fn) {
    vk::CommandBufferAllocateInfo allocInfo{*m_uploadPool,
                                            vk::CommandBufferLevel::ePrimary, 1};
    vk::raii::CommandBuffers buffers{m_device, allocInfo};
    vk::raii::CommandBuffer cmd = std::move(buffers.front());

    cmd.begin(vk::CommandBufferBeginInfo{
        vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    fn(cmd);
    cmd.end();

    vk::CommandBufferSubmitInfo cmdInfo{*cmd};
    vk::SubmitInfo2 submit{};
    submit.setCommandBufferInfos(cmdInfo);
    m_graphicsQueue.submit2(submit, *m_uploadFence);

    check(m_device.waitForFences(*m_uploadFence, VK_TRUE, UINT64_MAX),
          "waitForFences (immediate submit)");
    m_device.resetFences(*m_uploadFence);
}

} // namespace vve::gfx
