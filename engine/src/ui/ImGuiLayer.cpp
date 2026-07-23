#include "vve/ui/ImGuiLayer.hpp"

#include "vve/core/Window.hpp"
#include "vve/gfx/Context.hpp"
#include "vve/gfx/Swapchain.hpp"

#include <array>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <GLFW/glfw3.h>

namespace vve::ui {

ImGuiLayer::ImGuiLayer(core::Window& window, gfx::Context& ctx,
                       gfx::Swapchain& swapchain)
    : m_ctx(ctx) {
    // A generously sized descriptor pool for ImGui's textures.
    std::array<vk::DescriptorPoolSize, 1> sizes{
        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 64}};
    vk::DescriptorPoolCreateInfo poolInfo{
        vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 64};
    poolInfo.setPoolSizes(sizes);
    m_descriptorPool = vk::raii::DescriptorPool{ctx.device(), poolInfo};

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForVulkan(window.handle(), true);

    VkFormat colorFormat = static_cast<VkFormat>(swapchain.format());
    VkPipelineRenderingCreateInfo renderingInfo{
        VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.Instance = *ctx.instance();
    initInfo.PhysicalDevice = *ctx.physicalDevice();
    initInfo.Device = *ctx.device();
    initInfo.QueueFamily = *ctx.queueFamilies().graphics;
    initInfo.Queue = *ctx.graphicsQueue();
    initInfo.DescriptorPool = *m_descriptorPool;
    initInfo.MinImageCount = swapchain.imageCount();
    initInfo.ImageCount = swapchain.imageCount();
    initInfo.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineRenderingCreateInfo = renderingInfo;

    ImGui_ImplVulkan_Init(&initInfo);
    ImGui_ImplVulkan_CreateFontsTexture();
}

ImGuiLayer::~ImGuiLayer() {
    m_ctx.device().waitIdle();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void ImGuiLayer::beginFrame() {
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
}

void ImGuiLayer::render(vk::raii::CommandBuffer& cmd) {
    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(),
                                    static_cast<VkCommandBuffer>(*cmd));
}

} // namespace vve::ui
