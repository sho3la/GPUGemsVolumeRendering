// -----------------------------------------------------------------------------
// App 06 - Procedural volume rendering (GPU Gems §39.5.2).
//
// A coarse macrostructure volume is enriched at render time by a small tiling
// noise volume. Coordinate perturbation warps the lookup into wispy, cloud-like
// filaments; optical-property perturbation breaks up smooth interpolation by
// modulating opacity. Scrolling the noise over time animates the medium - the
// basis for real-time clouds, smoke and fire.
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
#include "vve/volume/SliceGeometryBuffers.hpp"
#include "vve/volume/SliceProxyGeometry.hpp"
#include "vve/volume/TransferFunction.hpp"
#include "vve/volume/VolumeData.hpp"
#include "vve/volume/VolumeSource.hpp"

#include <array>
#include <memory>
#include <random>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

using namespace vve;

namespace {
struct PushConstants {
    glm::mat4 mvp;
    glm::vec4 params;  // time, noiseScale, coordPerturb, propPerturb
    glm::vec4 params2; // opacityCorrection, flowSpeed
};
constexpr int kReferenceSlices = 256;
constexpr int kNoiseDim = 32;
constexpr size_t kMaxSliceVerts = 768 * 18;

// A small tiling RGBA noise volume, lightly box-blurred to hide the trilinear
// interpolation grid (as advised in §39.5.2).
std::vector<uint8_t> makeNoiseRGBA8(int dim, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    size_t n = static_cast<size_t>(dim) * dim * dim * 4;
    std::vector<float> raw(n);
    for (auto& v : raw) v = dist(rng);

    auto idx = [dim](int x, int y, int z, int c) {
        x = (x + dim) % dim; y = (y + dim) % dim; z = (z + dim) % dim;
        return ((static_cast<size_t>(z) * dim + y) * dim + x) * 4 + c;
    };
    std::vector<uint8_t> out(n);
    for (int z = 0; z < dim; ++z)
        for (int y = 0; y < dim; ++y)
            for (int x = 0; x < dim; ++x)
                for (int c = 0; c < 4; ++c) {
                    float sum = 0.0f;
                    for (int dz = -1; dz <= 1; ++dz)
                        for (int dy = -1; dy <= 1; ++dy)
                            for (int dx = -1; dx <= 1; ++dx)
                                sum += raw[idx(x + dx, y + dy, z + dz, c)];
                    out[idx(x, y, z, c)] =
                        static_cast<uint8_t>((sum / 27.0f) * 255.0f);
                }
    return out;
}
} // namespace

class ProceduralApp : public core::Application {
public:
    ProceduralApp() : core::Application(makeConfig()), m_arcball(m_camera) {}

private:
    static Config makeConfig() {
        Config c;
        c.window.title = "GPU Gems 39 - 06 Procedural Volume";
        return c;
    }

    void onInit() override {
        m_arcball.attach(window());
        m_arcball.setDistance(2.4f);
        m_sliceBuffers = std::make_unique<volume::SliceGeometryBuffers>(
            ctx(), gfx::Renderer::kFramesInFlight, kMaxSliceVerts);

        m_source.scanDirectory(core::dataDir());
        for (const auto& s : m_source.labels()) m_datasetNames.push_back(s.c_str());
        if (int r = m_source.firstRealIndex(); r >= 0) m_datasetIndex = r;

        createDescriptors();
        uploadTransferFunction(volume::TransferFunction::fire());
        uploadNoise();
        buildVolume(m_datasetIndex);
        createPipeline();
    }

    void createDescriptors() {
        auto sampler = [](uint32_t b) {
            return vk::DescriptorSetLayoutBinding{
                b, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment};
        };
        std::array<vk::DescriptorSetLayoutBinding, 3> bindings{
            sampler(0), sampler(1), sampler(2)};
        vk::DescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.setBindings(bindings);
        m_setLayout = vk::raii::DescriptorSetLayout{ctx().device(), layoutInfo};

        std::array<vk::DescriptorPoolSize, 1> sizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 3}};
        vk::DescriptorPoolCreateInfo poolInfo{
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 1};
        poolInfo.setPoolSizes(sizes);
        m_descPool = vk::raii::DescriptorPool{ctx().device(), poolInfo};

        vk::DescriptorSetLayout raw = *m_setLayout;
        vk::DescriptorSetAllocateInfo allocInfo{*m_descPool, 1, &raw};
        m_descSets = vk::raii::DescriptorSets{ctx().device(), allocInfo};

        gfx::Texture::Desc d1{};
        d1.extent = vk::Extent3D{256, 1, 1};
        d1.format = vk::Format::eR8G8B8A8Unorm;
        d1.type = vk::ImageType::e1D;
        m_tfTex = gfx::Texture(ctx(), d1);

        gfx::Texture::Desc dn{};
        dn.extent = vk::Extent3D{kNoiseDim, kNoiseDim, kNoiseDim};
        dn.format = vk::Format::eR8G8B8A8Unorm;
        dn.type = vk::ImageType::e3D;
        dn.addressMode = vk::SamplerAddressMode::eRepeat; // tiling
        m_noiseTex = gfx::Texture(ctx(), dn);
    }

    void writeSampler(uint32_t binding, vk::ImageView view, vk::Sampler sampler) {
        vk::DescriptorImageInfo info{sampler, view,
                                     vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet w{};
        w.dstSet = *m_descSets[0];
        w.dstBinding = binding;
        w.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        w.setImageInfo(info);
        ctx().device().updateDescriptorSets(w, {});
    }

    void uploadTransferFunction(const volume::TransferFunction& tf) {
        auto data = tf.bakeRGBA8(256, /*premultiply=*/false);
        m_tfTex.uploadFromData(ctx(), data.data(), data.size());
        writeSampler(1, m_tfTex.view(), m_tfTex.sampler());
    }

    void uploadNoise() {
        auto data = makeNoiseRGBA8(kNoiseDim, 90125);
        m_noiseTex.uploadFromData(ctx(), data.data(), data.size());
        writeSampler(2, m_noiseTex.view(), m_noiseTex.sampler());
    }

    void buildVolume(int index) {
        ctx().waitIdle();
        volume::VolumeData vol = m_source.create(index, 96);
        m_boxHalf = 0.5f * vol.normalizedExtent();

        gfx::Texture::Desc desc{};
        desc.extent = vk::Extent3D{static_cast<uint32_t>(vol.nx()),
                                   static_cast<uint32_t>(vol.ny()),
                                   static_cast<uint32_t>(vol.nz())};
        desc.format = vk::Format::eR8Unorm;
        desc.type = vk::ImageType::e3D;
        m_volumeTex = gfx::Texture(ctx(), desc);
        auto bytes = vol.toR8();
        m_volumeTex.uploadFromData(ctx(), bytes.data(), bytes.size());
        writeSampler(0, m_volumeTex.view(), m_volumeTex.sampler());
    }

    void createPipeline() {
        gfx::ShaderModule vert(ctx().device(), "proc.vert.spv");
        gfx::ShaderModule frag(ctx().device(), "proc.frag.spv");
        auto bindings = volume::SliceGeometryBuffers::bindings();
        auto attributes = volume::SliceGeometryBuffers::attributes();
        vk::PushConstantRange pcRange{
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sizeof(PushConstants)};

        gfx::GraphicsPipelineBuilder b;
        m_pipeline =
            b.setShaders(vert.stage(vk::ShaderStageFlagBits::eVertex),
                         frag.stage(vk::ShaderStageFlagBits::eFragment))
                .setVertexInput(bindings, attributes)
                .setColorFormat(swapchain().format())
                .setBlendAttachment(
                    gfx::GraphicsPipelineBuilder::blendPremultipliedOver())
                .addDescriptorSetLayout(*m_setLayout)
                .addPushConstantRange(pcRange)
                .build(ctx().device());
    }

    void onUpdate(float dt) override {
        if (m_animate) m_time += dt;
    }

    void onImGui() override {
        ImGui::Begin("Procedural Volume");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size())))
            buildVolume(m_datasetIndex);
        ImGui::SliderInt("Slices", &m_numSlices, 16, 768);
        ImGui::SeparatorText("Noise");
        ImGui::SliderFloat("Noise scale", &m_noiseScale, 0.5f, 8.0f);
        ImGui::SliderFloat("Coordinate perturb", &m_coordPerturb, 0.0f, 0.3f);
        ImGui::SliderFloat("Opacity perturb", &m_propPerturb, 0.0f, 2.0f);
        ImGui::Checkbox("Animate", &m_animate);
        ImGui::SameLine();
        ImGui::SliderFloat("Flow", &m_flowSpeed, 0.0f, 1.0f);
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
        pc.params = glm::vec4(m_time, m_noiseScale, m_coordPerturb, m_propPerturb);
        pc.params2 = glm::vec4(
            static_cast<float>(kReferenceSlices) / static_cast<float>(m_numSlices),
            m_flowSpeed, 0.0f, 0.0f);

        auto& cmd = *frame.cmd;
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_pipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *m_pipeline.layout,
                               0, *m_descSets[0], {});
        cmd.pushConstants<PushConstants>(
            *m_pipeline.layout,
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
    gfx::Texture m_tfTex;
    gfx::Texture m_noiseTex;
    gfx::Pipeline m_pipeline;
    vk::raii::DescriptorSetLayout m_setLayout{nullptr};
    vk::raii::DescriptorPool m_descPool{nullptr};
    vk::raii::DescriptorSets m_descSets{nullptr};

    glm::vec3 m_boxHalf{0.5f};
    int m_datasetIndex = 0;
    int m_numSlices = 256;
    float m_time = 0.0f;
    float m_noiseScale = 3.0f;
    float m_coordPerturb = 0.08f;
    float m_propPerturb = 0.5f;
    float m_flowSpeed = 0.3f;
    bool m_animate = true;
};

int main() {
    try {
        gfx::setShaderDir(std::string("shaders/") + VVE_SHADER_SUBDIR);
        ProceduralApp app;
        app.run();
    } catch (const std::exception& e) {
        vve::log::error("Fatal: %s", e.what());
        return 1;
    }
    return 0;
}
