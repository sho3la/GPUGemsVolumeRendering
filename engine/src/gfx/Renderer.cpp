#include "vve/gfx/Renderer.hpp"

#include "vve/gfx/Context.hpp"
#include "vve/gfx/Swapchain.hpp"

namespace vve::gfx {

Renderer::Renderer(Context& ctx, Swapchain& swapchain)
    : m_ctx(ctx), m_swapchain(swapchain) {
    createCommandBuffers();
    createSyncObjects();
}

void Renderer::createCommandBuffers() {
    m_commandPool = vk::raii::CommandPool{
        m_ctx.device(),
        vk::CommandPoolCreateInfo{
            vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
            *m_ctx.queueFamilies().graphics}};

    vk::CommandBufferAllocateInfo allocInfo{
        *m_commandPool, vk::CommandBufferLevel::ePrimary, kFramesInFlight};
    m_commandBuffers = vk::raii::CommandBuffers{m_ctx.device(), allocInfo};
}

void Renderer::createSyncObjects() {
    m_imageAvailable.clear();
    m_renderFinished.clear();
    m_inFlight.clear();

    for (uint32_t i = 0; i < kFramesInFlight; ++i) {
        m_imageAvailable.emplace_back(m_ctx.device(), vk::SemaphoreCreateInfo{});
        m_inFlight.emplace_back(
            m_ctx.device(),
            vk::FenceCreateInfo{vk::FenceCreateFlagBits::eSignaled});
    }
    for (uint32_t i = 0; i < m_swapchain.imageCount(); ++i) {
        m_renderFinished.emplace_back(m_ctx.device(), vk::SemaphoreCreateInfo{});
    }
}

Frame Renderer::beginFrame() {
    auto& device = m_ctx.device();

    check(device.waitForFences(*m_inFlight[m_currentFrame], VK_TRUE, UINT64_MAX),
          "waitForFences");

    uint32_t imageIndex = 0;
    vk::Result result = vk::Result::eSuccess;
    try {
        auto acquired =
            m_swapchain.acquireNextImage(*m_imageAvailable[m_currentFrame]);
        result = acquired.first;
        imageIndex = acquired.second;
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }

    if (result == vk::Result::eErrorOutOfDateKHR) {
        m_needsRecreate = true;
        return Frame{};
    }
    if (result != vk::Result::eSuccess && result != vk::Result::eSuboptimalKHR) {
        throw std::runtime_error("Failed to acquire swapchain image");
    }

    device.resetFences(*m_inFlight[m_currentFrame]);

    auto& cmd = m_commandBuffers[m_currentFrame];
    cmd.reset();
    cmd.begin(vk::CommandBufferBeginInfo{});

    return Frame{&cmd, imageIndex, m_currentFrame, true};
}

void Renderer::beginSwapchainPass(const Frame& frame,
                                  const std::array<float, 4>& clearColor) {
    auto& cmd = *frame.cmd;

    // UNDEFINED -> COLOR_ATTACHMENT_OPTIMAL
    vk::ImageMemoryBarrier2 toColor{};
    toColor.srcStageMask = vk::PipelineStageFlagBits2::eTopOfPipe;
    toColor.dstStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    toColor.srcAccessMask = {};
    toColor.dstAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    toColor.oldLayout = vk::ImageLayout::eUndefined;
    toColor.newLayout = vk::ImageLayout::eColorAttachmentOptimal;
    toColor.image = m_swapchain.image(frame.imageIndex);
    toColor.subresourceRange =
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    vk::DependencyInfo dep{};
    dep.setImageMemoryBarriers(toColor);
    cmd.pipelineBarrier2(dep);

    vk::RenderingAttachmentInfo colorAttachment{};
    colorAttachment.imageView = m_swapchain.view(frame.imageIndex);
    colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachment.clearValue.color =
        vk::ClearColorValue{clearColor[0], clearColor[1], clearColor[2],
                            clearColor[3]};

    vk::RenderingInfo renderingInfo{};
    renderingInfo.renderArea = vk::Rect2D{{0, 0}, m_swapchain.extent()};
    renderingInfo.layerCount = 1;
    renderingInfo.setColorAttachments(colorAttachment);
    cmd.beginRendering(renderingInfo);

    // Flip viewport so +Y is up (matches GLM clip conventions).
    vk::Viewport viewport{0.0f,
                          static_cast<float>(m_swapchain.extent().height),
                          static_cast<float>(m_swapchain.extent().width),
                          -static_cast<float>(m_swapchain.extent().height),
                          0.0f, 1.0f};
    cmd.setViewport(0, viewport);
    cmd.setScissor(0, vk::Rect2D{{0, 0}, m_swapchain.extent()});
}

void Renderer::endSwapchainPass(const Frame& frame) {
    auto& cmd = *frame.cmd;
    cmd.endRendering();

    vk::ImageMemoryBarrier2 toPresent{};
    toPresent.srcStageMask = vk::PipelineStageFlagBits2::eColorAttachmentOutput;
    toPresent.dstStageMask = vk::PipelineStageFlagBits2::eBottomOfPipe;
    toPresent.srcAccessMask = vk::AccessFlagBits2::eColorAttachmentWrite;
    toPresent.dstAccessMask = {};
    toPresent.oldLayout = vk::ImageLayout::eColorAttachmentOptimal;
    toPresent.newLayout = vk::ImageLayout::ePresentSrcKHR;
    toPresent.image = m_swapchain.image(frame.imageIndex);
    toPresent.subresourceRange =
        vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};

    vk::DependencyInfo dep{};
    dep.setImageMemoryBarriers(toPresent);
    cmd.pipelineBarrier2(dep);
}

void Renderer::endFrame(const Frame& frame) {
    auto& cmd = *frame.cmd;
    cmd.end();

    vk::CommandBufferSubmitInfo cmdInfo{*cmd};

    vk::SemaphoreSubmitInfo waitInfo{
        *m_imageAvailable[frame.frameIndex], 0,
        vk::PipelineStageFlagBits2::eColorAttachmentOutput};
    vk::SemaphoreSubmitInfo signalInfo{
        *m_renderFinished[frame.imageIndex], 0,
        vk::PipelineStageFlagBits2::eAllGraphics};

    vk::SubmitInfo2 submit{};
    submit.setWaitSemaphoreInfos(waitInfo);
    submit.setSignalSemaphoreInfos(signalInfo);
    submit.setCommandBufferInfos(cmdInfo);
    m_ctx.graphicsQueue().submit2(submit, *m_inFlight[frame.frameIndex]);

    vk::PresentInfoKHR present{};
    vk::Semaphore waitSem = *m_renderFinished[frame.imageIndex];
    present.setWaitSemaphores(waitSem);
    vk::SwapchainKHR sc = m_swapchain.handle();
    present.setSwapchains(sc);
    present.pImageIndices = &frame.imageIndex;

    vk::Result result = vk::Result::eSuccess;
    try {
        result = m_ctx.presentQueue().presentKHR(present);
    } catch (const vk::OutOfDateKHRError&) {
        result = vk::Result::eErrorOutOfDateKHR;
    }
    if (result == vk::Result::eErrorOutOfDateKHR ||
        result == vk::Result::eSuboptimalKHR) {
        m_needsRecreate = true;
    }

    m_currentFrame = (m_currentFrame + 1) % kFramesInFlight;
}

} // namespace vve::gfx
