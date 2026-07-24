#include "vve/gfx/PipelineBuilder.hpp"

#include <array>

namespace vve::gfx {

GraphicsPipelineBuilder::GraphicsPipelineBuilder() = default;

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setShaders(
    vk::PipelineShaderStageCreateInfo vert,
    vk::PipelineShaderStageCreateInfo frag) {
    m_vert = vert;
    m_frag = frag;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setVertexInput(
    const std::vector<vk::VertexInputBindingDescription>& bindings,
    const std::vector<vk::VertexInputAttributeDescription>& attributes) {
    m_bindings = bindings;
    m_attributes = attributes;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setTopology(
    vk::PrimitiveTopology topology) {
    m_topology = topology;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setPolygonMode(
    vk::PolygonMode mode) {
    m_polygonMode = mode;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setCullMode(
    vk::CullModeFlags mode) {
    m_cullMode = mode;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setColorFormat(
    vk::Format format) {
    m_colorFormat = format;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::setBlendAttachment(
    const vk::PipelineColorBlendAttachmentState& blend) {
    m_blend = blend;
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::addDescriptorSetLayout(
    vk::DescriptorSetLayout set) {
    m_setLayouts.push_back(set);
    return *this;
}

GraphicsPipelineBuilder& GraphicsPipelineBuilder::addPushConstantRange(
    vk::PushConstantRange range) {
    m_pushConstants.push_back(range);
    return *this;
}

vk::PipelineColorBlendAttachmentState GraphicsPipelineBuilder::blendNone() {
    vk::PipelineColorBlendAttachmentState b{};
    b.blendEnable = VK_FALSE;
    b.colorWriteMask = vk::ColorComponentFlagBits::eR |
                       vk::ColorComponentFlagBits::eG |
                       vk::ColorComponentFlagBits::eB |
                       vk::ColorComponentFlagBits::eA;
    return b;
}

vk::PipelineColorBlendAttachmentState
GraphicsPipelineBuilder::blendPremultipliedOver() {
    vk::PipelineColorBlendAttachmentState b = blendNone();
    b.blendEnable = VK_TRUE;
    b.srcColorBlendFactor = vk::BlendFactor::eOne;
    b.dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    b.colorBlendOp = vk::BlendOp::eAdd;
    b.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    b.dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha;
    b.alphaBlendOp = vk::BlendOp::eAdd;
    return b;
}

vk::PipelineColorBlendAttachmentState GraphicsPipelineBuilder::blendUnder() {
    vk::PipelineColorBlendAttachmentState b = blendNone();
    b.blendEnable = VK_TRUE;
    b.srcColorBlendFactor = vk::BlendFactor::eOneMinusDstAlpha;
    b.dstColorBlendFactor = vk::BlendFactor::eOne;
    b.colorBlendOp = vk::BlendOp::eAdd;
    b.srcAlphaBlendFactor = vk::BlendFactor::eOneMinusDstAlpha;
    b.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    b.alphaBlendOp = vk::BlendOp::eAdd;
    return b;
}

vk::PipelineColorBlendAttachmentState GraphicsPipelineBuilder::blendMax() {
    vk::PipelineColorBlendAttachmentState b = blendNone();
    b.blendEnable = VK_TRUE;
    b.srcColorBlendFactor = vk::BlendFactor::eOne;
    b.dstColorBlendFactor = vk::BlendFactor::eOne;
    b.colorBlendOp = vk::BlendOp::eMax;
    b.srcAlphaBlendFactor = vk::BlendFactor::eOne;
    b.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    b.alphaBlendOp = vk::BlendOp::eMax;
    return b;
}

vk::PipelineColorBlendAttachmentState GraphicsPipelineBuilder::blendMultiply() {
    vk::PipelineColorBlendAttachmentState b = blendNone();
    b.blendEnable = VK_TRUE;
    b.srcColorBlendFactor = vk::BlendFactor::eZero;
    b.dstColorBlendFactor = vk::BlendFactor::eSrcColor; // Cd' = Cd * Cs
    b.colorBlendOp = vk::BlendOp::eAdd;
    b.srcAlphaBlendFactor = vk::BlendFactor::eZero;
    b.dstAlphaBlendFactor = vk::BlendFactor::eOne;
    b.alphaBlendOp = vk::BlendOp::eAdd;
    return b;
}

Pipeline GraphicsPipelineBuilder::build(vk::raii::Device& device) {
    Pipeline result;

    vk::PipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.setSetLayouts(m_setLayouts);
    layoutInfo.setPushConstantRanges(m_pushConstants);
    result.layout = vk::raii::PipelineLayout{device, layoutInfo};

    std::array<vk::PipelineShaderStageCreateInfo, 2> stages{m_vert, m_frag};

    vk::PipelineVertexInputStateCreateInfo vertexInput{};
    vertexInput.setVertexBindingDescriptions(m_bindings);
    vertexInput.setVertexAttributeDescriptions(m_attributes);

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = m_topology;

    vk::PipelineViewportStateCreateInfo viewport{};
    viewport.viewportCount = 1;
    viewport.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo raster{};
    raster.polygonMode = m_polygonMode;
    raster.cullMode = m_cullMode;
    raster.frontFace = vk::FrontFace::eCounterClockwise;
    raster.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisample{};
    multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = VK_FALSE;
    depthStencil.depthWriteEnable = VK_FALSE;

    vk::PipelineColorBlendStateCreateInfo colorBlend{};
    colorBlend.setAttachments(m_blend);

    std::array<vk::DynamicState, 2> dynamicStates{
        vk::DynamicState::eViewport, vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.setDynamicStates(dynamicStates);

    vk::PipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.setColorAttachmentFormats(m_colorFormat);

    vk::GraphicsPipelineCreateInfo pipelineInfo{};
    pipelineInfo.pNext = &renderingInfo;
    pipelineInfo.setStages(stages);
    pipelineInfo.pVertexInputState = &vertexInput;
    pipelineInfo.pInputAssemblyState = &inputAssembly;
    pipelineInfo.pViewportState = &viewport;
    pipelineInfo.pRasterizationState = &raster;
    pipelineInfo.pMultisampleState = &multisample;
    pipelineInfo.pDepthStencilState = &depthStencil;
    pipelineInfo.pColorBlendState = &colorBlend;
    pipelineInfo.pDynamicState = &dynamicState;
    pipelineInfo.layout = *result.layout;

    result.pipeline = vk::raii::Pipeline{device, nullptr, pipelineInfo};
    return result;
}

} // namespace vve::gfx
