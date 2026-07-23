// -----------------------------------------------------------------------------
// App 02 - Transfer functions (GPU Gems §39.4.3).
//
// Post-classification with a 1D transfer function (value -> color/opacity) and a
// 2D transfer function (value + gradient magnitude -> color/opacity) for
// boundary emphasis. Includes an interactive control-point editor, colormap
// presets, and opacity correction (Equation 3) so image intensity stays
// constant as the slice count changes.
// -----------------------------------------------------------------------------
#include "vve/core/Application.hpp"
#include "vve/core/Log.hpp"
#include "vve/core/Paths.hpp"
#include "vve/gfx/Context.hpp"
#include "vve/gfx/PipelineBuilder.hpp"
#include "vve/gfx/Renderer.hpp"
#include "vve/gfx/ShaderModule.hpp"
#include "vve/gfx/Swapchain.hpp"
#include "vve/gfx/Texture.hpp"
#include "vve/scene/ArcballController.hpp"
#include "vve/scene/Camera.hpp"
#include "vve/volume/GradientComputer.hpp"
#include "vve/volume/SliceGeometryBuffers.hpp"
#include "vve/volume/SliceProxyGeometry.hpp"
#include "vve/volume/TransferFunction.hpp"
#include "vve/volume/VolumeData.hpp"
#include "vve/volume/VolumeSource.hpp"

#include <array>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

using namespace vve;

namespace {
struct PushConstants {
    glm::mat4 mvp;
    float opacityCorrection;
};
constexpr int kReferenceSlices = 256;
constexpr int kTF1DWidth = 256;
constexpr int kTF2DWidth = 256;
constexpr int kTF2DHeight = 64;
constexpr size_t kMaxSliceVerts = 768 * 18;
} // namespace

class TransferFunctionApp : public core::Application {
public:
    TransferFunctionApp()
        : core::Application(makeConfig()), m_arcball(m_camera) {}

private:
    static Config makeConfig() {
        Config c;
        c.window.title = "GPU Gems 39 - 02 Transfer Functions";
        return c;
    }

    void onInit() override {
        m_arcball.attach(window());
        m_arcball.setDistance(2.2f);

        m_tf = volume::TransferFunction::fire();

        m_sliceBuffers = std::make_unique<volume::SliceGeometryBuffers>(
            ctx(), gfx::Renderer::kFramesInFlight, kMaxSliceVerts);

        m_source.scanDirectory(core::dataDir());
        for (const auto& s : m_source.labels()) m_datasetNames.push_back(s.c_str());
        if (int r = m_source.firstRealIndex(); r >= 0) m_datasetIndex = r;

        createDescriptors();
        createTransferTextures();
        buildVolume(m_datasetIndex);
        rebakeTransferFunctions();
        createPipelines();
    }

    void createDescriptors() {
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{
            vk::DescriptorSetLayoutBinding{
                0, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment},
            vk::DescriptorSetLayoutBinding{
                1, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment}};
        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);
        m_setLayout = vk::raii::DescriptorSetLayout{ctx().device(), layoutInfo};

        std::array<vk::DescriptorPoolSize, 1> sizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 4}};
        vk::DescriptorPoolCreateInfo poolInfo{
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 2};
        poolInfo.setPoolSizes(sizes);
        m_descPool = vk::raii::DescriptorPool{ctx().device(), poolInfo};

        std::array<vk::DescriptorSetLayout, 2> layouts{*m_setLayout, *m_setLayout};
        vk::DescriptorSetAllocateInfo allocInfo{*m_descPool, layouts};
        m_descSets = vk::raii::DescriptorSets{ctx().device(), allocInfo};
    }

    void createTransferTextures() {
        gfx::Texture::Desc d1{};
        d1.extent = vk::Extent3D{kTF1DWidth, 1, 1};
        d1.format = vk::Format::eR8G8B8A8Unorm;
        d1.type = vk::ImageType::e1D;
        m_tf1dTex = gfx::Texture(ctx(), d1);

        gfx::Texture::Desc d2{};
        d2.extent = vk::Extent3D{kTF2DWidth, kTF2DHeight, 1};
        d2.format = vk::Format::eR8G8B8A8Unorm;
        d2.type = vk::ImageType::e2D;
        m_tf2dTex = gfx::Texture(ctx(), d2);
    }

    void buildVolume(int index) {
        ctx().waitIdle();
        volume::VolumeData vol = m_source.create(index, 128);
        m_boxHalf = 0.5f * vol.normalizedExtent();

        gfx::Texture::Desc desc{};
        desc.extent = vk::Extent3D{static_cast<uint32_t>(vol.nx()),
                                   static_cast<uint32_t>(vol.ny()),
                                   static_cast<uint32_t>(vol.nz())};
        desc.format = vk::Format::eR8G8Unorm; // r = value, g = |gradient|
        desc.type = vk::ImageType::e3D;
        m_volumeTex = gfx::Texture(ctx(), desc);
        auto bytes = volume::GradientComputer::valueAndGradientMagnitudeRG8(vol);
        m_volumeTex.uploadFromData(ctx(), bytes.data(), bytes.size());

        // Bind the volume into both descriptor sets (binding 0).
        for (auto& set : m_descSets) {
            auto info = m_volumeTex.descriptor();
            vk::WriteDescriptorSet w{};
            w.dstSet = *set;
            w.dstBinding = 0;
            w.descriptorType = vk::DescriptorType::eCombinedImageSampler;
            w.setImageInfo(info);
            ctx().device().updateDescriptorSets(w, {});
        }
    }

    void rebakeTransferFunctions() {
        ctx().waitIdle();

        // 1D transfer function (straight color; opacity corrected in shader).
        auto tf1d = m_tf.bakeRGBA8(kTF1DWidth, /*premultiply=*/false);
        m_tf1dTex.uploadFromData(ctx(), tf1d.data(), tf1d.size());

        // 2D transfer function: reuse the 1D color ramp, modulate opacity by
        // gradient magnitude so material boundaries stand out (§39.4.3).
        std::vector<uint8_t> tf2d(static_cast<size_t>(kTF2DWidth) * kTF2DHeight * 4);
        for (int gy = 0; gy < kTF2DHeight; ++gy) {
            float g = static_cast<float>(gy) / (kTF2DHeight - 1);
            for (int vx = 0; vx < kTF2DWidth; ++vx) {
                size_t src = static_cast<size_t>(vx) * 4;
                float baseA = tf1d[src + 3] / 255.0f;
                float a = baseA * ((1.0f - m_gradEmphasis) + m_gradEmphasis * g);
                size_t dst = (static_cast<size_t>(gy) * kTF2DWidth + vx) * 4;
                tf2d[dst + 0] = tf1d[src + 0];
                tf2d[dst + 1] = tf1d[src + 1];
                tf2d[dst + 2] = tf1d[src + 2];
                tf2d[dst + 3] = static_cast<uint8_t>(a * 255.0f);
            }
        }
        m_tf2dTex.uploadFromData(ctx(), tf2d.data(), tf2d.size());

        // Point the transfer-function binding at the right image per set.
        auto info1d = m_tf1dTex.descriptor();
        vk::WriteDescriptorSet w1{};
        w1.dstSet = *m_descSets[0];
        w1.dstBinding = 1;
        w1.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        w1.setImageInfo(info1d);

        auto info2d = m_tf2dTex.descriptor();
        vk::WriteDescriptorSet w2{};
        w2.dstSet = *m_descSets[1];
        w2.dstBinding = 1;
        w2.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        w2.setImageInfo(info2d);

        ctx().device().updateDescriptorSets({w1, w2}, {});
    }

    void createPipelines() {
        gfx::ShaderModule vert(ctx().device(), "slice.vert.spv");
        gfx::ShaderModule frag1d(ctx().device(), "tf1d.frag.spv");
        gfx::ShaderModule frag2d(ctx().device(), "tf2d.frag.spv");

        auto bindings = volume::SliceGeometryBuffers::bindings();
        auto attributes = volume::SliceGeometryBuffers::attributes();
        vk::PushConstantRange pcRange{
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sizeof(PushConstants)};

        auto makePipeline = [&](const gfx::ShaderModule& frag) {
            gfx::GraphicsPipelineBuilder b;
            return b.setShaders(
                        vert.stage(vk::ShaderStageFlagBits::eVertex),
                        frag.stage(vk::ShaderStageFlagBits::eFragment))
                .setVertexInput(bindings, attributes)
                .setColorFormat(swapchain().format())
                .setBlendAttachment(
                    gfx::GraphicsPipelineBuilder::blendPremultipliedOver())
                .addDescriptorSetLayout(*m_setLayout)
                .addPushConstantRange(pcRange)
                .build(ctx().device());
        };
        m_pipeline1D = makePipeline(frag1d);
        m_pipeline2D = makePipeline(frag2d);
    }

    void onImGui() override {
        ImGui::Begin("Transfer Functions");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size()))) {
            buildVolume(m_datasetIndex);
        }
        ImGui::Checkbox("2D transfer function (value + |gradient|)", &m_use2D);
        ImGui::SliderInt("Slices", &m_numSlices, 16, 768);

        bool dirty = false;
        if (ImGui::Button("Fire")) { m_tf = volume::TransferFunction::fire(); dirty = true; }
        ImGui::SameLine();
        if (ImGui::Button("Grayscale")) { m_tf = volume::TransferFunction::grayscaleRamp(); dirty = true; }
        ImGui::SameLine();
        if (ImGui::Button("Cool-Warm")) { m_tf = volume::TransferFunction::coolWarm(); dirty = true; }

        if (m_use2D) {
            if (ImGui::SliderFloat("Gradient emphasis", &m_gradEmphasis, 0.0f, 1.0f))
                dirty = true;
        }

        ImGui::SeparatorText("Control points");
        auto& stops = m_tf.stops();
        for (size_t i = 0; i < stops.size(); ++i) {
            ImGui::PushID(static_cast<int>(i));
            if (ImGui::SliderFloat("pos", &stops[i].t, 0.0f, 1.0f)) dirty = true;
            if (ImGui::ColorEdit3("color", &stops[i].color.x)) dirty = true;
            if (ImGui::SliderFloat("opacity", &stops[i].opacity, 0.0f, 1.0f)) dirty = true;
            ImGui::PopID();
            ImGui::Separator();
        }

        if (dirty) rebakeTransferFunctions();

        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::End();
    }

    void onRender(gfx::Frame& frame) override {
        auto verts = volume::SliceProxyGeometry::generate(
            -m_boxHalf, m_boxHalf, m_camera.forward(), m_numSlices,
            /*reverse=*/true);
        uint32_t count = m_sliceBuffers->upload(frame.frameIndex, verts);
        if (count == 0) return;

        float aspect = static_cast<float>(swapchain().extent().width) /
                       static_cast<float>(swapchain().extent().height);
        PushConstants pc{};
        pc.mvp = m_camera.projection(aspect) * m_camera.view();
        pc.opacityCorrection =
            static_cast<float>(kReferenceSlices) / static_cast<float>(m_numSlices);

        gfx::Pipeline& pipeline = m_use2D ? m_pipeline2D : m_pipeline1D;
        vk::DescriptorSet set = m_use2D ? *m_descSets[1] : *m_descSets[0];

        auto& cmd = *frame.cmd;
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipeline.layout,
                               0, set, {});
        cmd.pushConstants<PushConstants>(
            *pipeline.layout,
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, pc);
        vk::DeviceSize offset = 0;
        vk::Buffer vb = m_sliceBuffers->buffer(frame.frameIndex);
        cmd.bindVertexBuffers(0, vb, offset);
        cmd.draw(count, 1, 0, 0);
    }

    scene::Camera m_camera;
    scene::ArcballController m_arcball;
    volume::VolumeSource m_source;
    std::vector<const char*> m_datasetNames;

    std::unique_ptr<volume::SliceGeometryBuffers> m_sliceBuffers;
    gfx::Texture m_volumeTex;
    gfx::Texture m_tf1dTex;
    gfx::Texture m_tf2dTex;
    volume::TransferFunction m_tf;

    gfx::Pipeline m_pipeline1D;
    gfx::Pipeline m_pipeline2D;
    vk::raii::DescriptorSetLayout m_setLayout{nullptr};
    vk::raii::DescriptorPool m_descPool{nullptr};
    vk::raii::DescriptorSets m_descSets{nullptr};

    glm::vec3 m_boxHalf{0.5f};
    int m_datasetIndex = 0;
    int m_numSlices = 256;
    bool m_use2D = false;
    float m_gradEmphasis = 0.7f;
};

int main() {
    try {
        gfx::setShaderDir(std::string("shaders/") + VVE_SHADER_SUBDIR);
        TransferFunctionApp app;
        app.run();
    } catch (const std::exception& e) {
        vve::log::error("Fatal: %s", e.what());
        return 1;
    }
    return 0;
}
