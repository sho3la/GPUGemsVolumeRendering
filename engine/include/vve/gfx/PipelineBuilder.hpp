#pragma once

#include "vve/gfx/VulkanCommon.hpp"

#include <vector>

namespace vve::gfx {

// The layout + pipeline pair produced by the builder.
struct Pipeline {
    vk::raii::PipelineLayout layout{nullptr};
    vk::raii::Pipeline pipeline{nullptr};
};

// Fluent builder for a dynamic-rendering graphics pipeline. Keeps the verbose
// Vulkan pipeline setup in one reusable place (SRP) so apps declare only what
// differs between techniques.
class GraphicsPipelineBuilder {
public:
    GraphicsPipelineBuilder();

    GraphicsPipelineBuilder& setShaders(vk::PipelineShaderStageCreateInfo vert,
                                        vk::PipelineShaderStageCreateInfo frag);
    GraphicsPipelineBuilder& setVertexInput(
        const std::vector<vk::VertexInputBindingDescription>& bindings,
        const std::vector<vk::VertexInputAttributeDescription>& attributes);
    GraphicsPipelineBuilder& setTopology(vk::PrimitiveTopology topology);
    GraphicsPipelineBuilder& setPolygonMode(vk::PolygonMode mode);
    GraphicsPipelineBuilder& setCullMode(vk::CullModeFlags mode);
    GraphicsPipelineBuilder& setColorFormat(vk::Format format);
    GraphicsPipelineBuilder& setBlendAttachment(
        const vk::PipelineColorBlendAttachmentState& blend);
    GraphicsPipelineBuilder& addDescriptorSetLayout(vk::DescriptorSetLayout set);
    GraphicsPipelineBuilder& addPushConstantRange(vk::PushConstantRange range);

    Pipeline build(vk::raii::Device& device);

    // Common blend presets from the chapter's compositing operators.
    static vk::PipelineColorBlendAttachmentState blendNone();
    // Premultiplied "over": Cs*1 + Cd*(1 - As). (§39.4.3, Equation 5)
    static vk::PipelineColorBlendAttachmentState blendPremultipliedOver();
    // "Under" for front-to-back: Cs*(1 - Ad) + Cd*1. (§39.4.3, Equation 6)
    static vk::PipelineColorBlendAttachmentState blendUnder();
    // Multiplicative: Cd' = Cd * Cs. Per-channel light attenuation for the
    // translucency light buffer (§39.5.1).
    static vk::PipelineColorBlendAttachmentState blendMultiply();
    // Maximum: Cd' = max(Cs, Cd). Order-independent, used for MIP rendering.
    static vk::PipelineColorBlendAttachmentState blendMax();

private:
    vk::PipelineShaderStageCreateInfo m_vert{};
    vk::PipelineShaderStageCreateInfo m_frag{};
    std::vector<vk::VertexInputBindingDescription> m_bindings;
    std::vector<vk::VertexInputAttributeDescription> m_attributes;
    vk::PrimitiveTopology m_topology = vk::PrimitiveTopology::eTriangleList;
    vk::PolygonMode m_polygonMode = vk::PolygonMode::eFill;
    vk::CullModeFlags m_cullMode = vk::CullModeFlagBits::eNone;
    vk::Format m_colorFormat = vk::Format::eB8G8R8A8Unorm;
    vk::PipelineColorBlendAttachmentState m_blend = blendNone();
    std::vector<vk::DescriptorSetLayout> m_setLayouts;
    std::vector<vk::PushConstantRange> m_pushConstants;
};

} // namespace vve::gfx
