// -----------------------------------------------------------------------------
// App 01 - Basic emissive volume rendering.
//
// Implements the foundational technique of GPU Gems §39.2-39.3: upload a scalar
// field as a 3D texture, generate view-aligned proxy geometry each frame
// (§39.4.2, Algorithm 39-2) and composite the slices back-to-front with the
// "over" operator (§39.4.3). Each sample emits density * emissive color
// (Listing 39-1) - no transfer function or lighting yet.
// -----------------------------------------------------------------------------
#include "vve/core/Application.hpp"
#include "vve/core/Log.hpp"
#include "vve/core/Paths.hpp"
#include "vve/gfx/Buffer.hpp"
#include "vve/gfx/Context.hpp"
#include "vve/gfx/PipelineBuilder.hpp"
#include "vve/gfx/Renderer.hpp"
#include "vve/gfx/ShaderModule.hpp"
#include "vve/gfx/Swapchain.hpp"
#include "vve/gfx/Texture.hpp"
#include "vve/scene/ArcballController.hpp"
#include "vve/scene/Camera.hpp"
#include "vve/volume/SliceProxyGeometry.hpp"
#include "vve/volume/VolumeData.hpp"
#include "vve/volume/VolumeSource.hpp"

#include <array>
#include <cstring>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

using namespace vve;

namespace {
struct PushConstants {
    glm::mat4 mvp;
    glm::vec4 emissive;
    float densityScale;
};

constexpr int kMaxSlices = 768;
constexpr int kMaxVertsPerSlice = 18; // convex hull of a box slice fans to <=4 tris
} // namespace

class BasicEmissiveApp : public core::Application {
public:
    BasicEmissiveApp()
        : core::Application(makeConfig()), m_arcball(m_camera) {}

private:
    static Config makeConfig() {
        Config c;
        c.window.title = "GPU Gems 39 - 01 Basic Emissive Volume";
        return c;
    }

    void onInit() override {
        m_arcball.attach(window());
        m_arcball.setDistance(2.2f);

        m_source.scanDirectory(core::dataDir());
        for (const auto& s : m_source.labels()) m_datasetNames.push_back(s.c_str());
        if (int r = m_source.firstRealIndex(); r >= 0) m_datasetIndex = r;

        createDescriptors();
        buildVolume(m_datasetIndex);
        createPipeline();

        // Per-frame slice buffers (host-visible, persistently mapped).
        const vk::DeviceSize capacity =
            sizeof(volume::SliceProxyGeometry::Vertex) *
            static_cast<vk::DeviceSize>(kMaxSlices) * kMaxVertsPerSlice;
        for (auto& buf : m_sliceBuffers) {
            buf = gfx::Buffer(ctx(), capacity,
                              vk::BufferUsageFlagBits::eVertexBuffer,
                              VMA_MEMORY_USAGE_AUTO,
                              VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                                  VMA_ALLOCATION_CREATE_MAPPED_BIT);
        }
    }

    void createDescriptors() {
        vk::DescriptorSetLayoutBinding binding{
            0, vk::DescriptorType::eCombinedImageSampler, 1,
            vk::ShaderStageFlagBits::eFragment};
        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(binding);
        m_setLayout = vk::raii::DescriptorSetLayout{ctx().device(), layoutInfo};

        std::array<vk::DescriptorPoolSize, 1> sizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1}};
        vk::DescriptorPoolCreateInfo poolInfo{
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1};
        poolInfo.setPoolSizes(sizes);
        m_descPool = vk::raii::DescriptorPool{ctx().device(), poolInfo};

        vk::DescriptorSetLayout raw = *m_setLayout;
        vk::DescriptorSetAllocateInfo allocInfo{*m_descPool, 1, &raw};
        m_descSets = vk::raii::DescriptorSets{ctx().device(), allocInfo};
    }

    void buildVolume(int index) {
        ctx().waitIdle();
        m_volume = m_source.create(index, 128);

        gfx::Texture::Desc desc{};
        desc.extent = vk::Extent3D{static_cast<uint32_t>(m_volume.nx()),
                                   static_cast<uint32_t>(m_volume.ny()),
                                   static_cast<uint32_t>(m_volume.nz())};
        desc.format = vk::Format::eR8Unorm;
        desc.type = vk::ImageType::e3D;
        m_volumeTex = gfx::Texture(ctx(), desc);
        auto bytes = m_volume.toR8();
        m_volumeTex.uploadFromData(ctx(), bytes.data(), bytes.size());

        auto imageInfo = m_volumeTex.descriptor();
        vk::WriteDescriptorSet write{};
        write.dstSet = *m_descSets[0];
        write.dstBinding = 0;
        write.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        write.setImageInfo(imageInfo);
        ctx().device().updateDescriptorSets(write, {});

        m_boxHalf = 0.5f * m_volume.normalizedExtent();
    }

    void createPipeline() {
        gfx::ShaderModule vert(ctx().device(), "slice.vert.spv");
        gfx::ShaderModule frag(ctx().device(), "slice.frag.spv");

        std::vector<vk::VertexInputBindingDescription> bindings{
            {0, sizeof(volume::SliceProxyGeometry::Vertex),
             vk::VertexInputRate::eVertex}};
        std::vector<vk::VertexInputAttributeDescription> attributes{
            {0, 0, vk::Format::eR32G32B32Sfloat,
             offsetof(volume::SliceProxyGeometry::Vertex, position)},
            {1, 0, vk::Format::eR32G32B32Sfloat,
             offsetof(volume::SliceProxyGeometry::Vertex, uvw)}};

        vk::PushConstantRange pcRange{
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sizeof(PushConstants)};

        gfx::GraphicsPipelineBuilder builder;
        m_pipeline =
            builder.setShaders(vert.stage(vk::ShaderStageFlagBits::eVertex),
                               frag.stage(vk::ShaderStageFlagBits::eFragment))
                .setVertexInput(bindings, attributes)
                .setTopology(vk::PrimitiveTopology::eTriangleList)
                .setColorFormat(swapchain().format())
                .setBlendAttachment(
                    gfx::GraphicsPipelineBuilder::blendPremultipliedOver())
                .addDescriptorSetLayout(*m_setLayout)
                .addPushConstantRange(pcRange)
                .build(ctx().device());
    }

    void onImGui() override {
        ImGui::Begin("Volume (Basic Emissive)");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size()))) {
            buildVolume(m_datasetIndex);
        }
        ImGui::SliderInt("Slices", &m_numSlices, 16, kMaxSlices);
        ImGui::ColorEdit3("Emissive", &m_emissive.x);
        ImGui::SliderFloat("Density scale", &m_densityScale, 0.0f, 4.0f);
        ImGui::Separator();
        ImGui::Text("Left-drag: orbit   Scroll: zoom");
        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::End();
    }

    void onRender(gfx::Frame& frame) override {
        // Generate proxy geometry along the current view direction, ordered
        // back-to-front for the "over" operator.
        glm::vec3 viewDir = m_camera.forward();
        auto verts = volume::SliceProxyGeometry::generate(
            -m_boxHalf, m_boxHalf, viewDir, m_numSlices, /*reverse=*/true);
        if (verts.empty()) return;

        auto& buffer = m_sliceBuffers[frame.frameIndex];
        size_t bytes = verts.size() * sizeof(verts[0]);
        std::memcpy(buffer.mapped(), verts.data(), bytes);

        float aspect = static_cast<float>(swapchain().extent().width) /
                       static_cast<float>(swapchain().extent().height);
        PushConstants pc{};
        pc.mvp = m_camera.projection(aspect) * m_camera.view();
        pc.emissive = glm::vec4(m_emissive, 1.0f);
        pc.densityScale = m_densityScale;

        auto& cmd = *frame.cmd;
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_pipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               *m_pipeline.layout, 0, *m_descSets[0], {});
        cmd.pushConstants<PushConstants>(
            *m_pipeline.layout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, pc);
        vk::DeviceSize offset = 0;
        vk::Buffer vb = buffer.handle();
        cmd.bindVertexBuffers(0, vb, offset);
        cmd.draw(static_cast<uint32_t>(verts.size()), 1, 0, 0);
    }

    // Scene ------------------------------------------------------------------
    scene::Camera m_camera;
    scene::ArcballController m_arcball;
    volume::VolumeSource m_source;
    std::vector<const char*> m_datasetNames;

    // GPU resources ----------------------------------------------------------
    volume::VolumeData m_volume;
    gfx::Texture m_volumeTex;
    gfx::Pipeline m_pipeline;
    vk::raii::DescriptorSetLayout m_setLayout{nullptr};
    vk::raii::DescriptorPool m_descPool{nullptr};
    vk::raii::DescriptorSets m_descSets{nullptr};
    std::array<gfx::Buffer, gfx::Renderer::kFramesInFlight> m_sliceBuffers;

    // Parameters -------------------------------------------------------------
    glm::vec3 m_boxHalf{0.5f};
    int m_datasetIndex = 0;
    int m_numSlices = 256;
    glm::vec3 m_emissive{0.9f, 0.6f, 0.3f};
    float m_densityScale = 1.0f;
};

int main() {
    try {
        gfx::setShaderDir(std::string("shaders/") + VVE_SHADER_SUBDIR);
        BasicEmissiveApp app;
        app.run();
    } catch (const std::exception& e) {
        vve::log::error("Fatal: %s", e.what());
        return 1;
    }
    return 0;
}
