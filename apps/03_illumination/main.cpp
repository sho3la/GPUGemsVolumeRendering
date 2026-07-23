// -----------------------------------------------------------------------------
// App 03 - Volume illumination (GPU Gems §39.4.3).
//
// Adds a Blinn-Phong local-illumination model (Equation 4) to the sliced volume
// renderer. The surface normal is the normalized gradient of the scalar field,
// evaluated in the fragment shader by central differences (Equation 2). Shading
// is confined to high-gradient regions so that boundaries gain relief while
// homogeneous interiors remain governed by the transfer function.
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
#include <cmath>
#include <memory>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

using namespace vve;

namespace {
struct PushConstants {
    glm::mat4 mvp;
    glm::vec4 texelSize;
    glm::vec4 lightDir; // xyz dir, w = ambient
    glm::vec4 viewDir;  // xyz dir, w = shininess
    glm::vec4 params;   // x=kd, y=ks, z=shadeStrength, w=opacityCorrection
};
constexpr int kReferenceSlices = 256;
constexpr size_t kMaxSliceVerts = 768 * 18;
} // namespace

class IlluminationApp : public core::Application {
public:
    IlluminationApp() : core::Application(makeConfig()), m_arcball(m_camera) {}

private:
    static Config makeConfig() {
        Config c;
        c.window.title = "GPU Gems 39 - 03 Volume Illumination";
        return c;
    }

    void onInit() override {
        m_arcball.attach(window());
        m_arcball.setDistance(2.2f);

        m_sliceBuffers = std::make_unique<volume::SliceGeometryBuffers>(
            ctx(), gfx::Renderer::kFramesInFlight, kMaxSliceVerts);

        m_source.scanDirectory(core::dataDir());
        for (const auto& s : m_source.labels()) m_datasetNames.push_back(s.c_str());
        if (int r = m_source.firstRealIndex(); r >= 0) m_datasetIndex = r;

        createDescriptors();
        uploadTransferFunction(volume::TransferFunction::coolWarm());
        buildVolume(m_datasetIndex);
        createPipeline();
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
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 2}};
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
    }

    void uploadTransferFunction(const volume::TransferFunction& tf) {
        auto data = tf.bakeRGBA8(256, /*premultiply=*/false);
        m_tfTex.uploadFromData(ctx(), data.data(), data.size());

        auto info = m_tfTex.descriptor();
        vk::WriteDescriptorSet w{};
        w.dstSet = *m_descSets[0];
        w.dstBinding = 1;
        w.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        w.setImageInfo(info);
        ctx().device().updateDescriptorSets(w, {});
    }

    void buildVolume(int index) {
        ctx().waitIdle();
        volume::VolumeData vol = m_source.create(index, 128);
        m_boxHalf = 0.5f * vol.normalizedExtent();
        m_texelSize = glm::vec3(1.0f / vol.nx(), 1.0f / vol.ny(), 1.0f / vol.nz());

        gfx::Texture::Desc desc{};
        desc.extent = vk::Extent3D{static_cast<uint32_t>(vol.nx()),
                                   static_cast<uint32_t>(vol.ny()),
                                   static_cast<uint32_t>(vol.nz())};
        desc.format = vk::Format::eR8Unorm;
        desc.type = vk::ImageType::e3D;
        m_volumeTex = gfx::Texture(ctx(), desc);
        auto bytes = vol.toR8();
        m_volumeTex.uploadFromData(ctx(), bytes.data(), bytes.size());

        auto info = m_volumeTex.descriptor();
        vk::WriteDescriptorSet w{};
        w.dstSet = *m_descSets[0];
        w.dstBinding = 0;
        w.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        w.setImageInfo(info);
        ctx().device().updateDescriptorSets(w, {});
    }

    void createPipeline() {
        gfx::ShaderModule vert(ctx().device(), "lit.vert.spv");
        gfx::ShaderModule frag(ctx().device(), "lit.frag.spv");

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

    void onImGui() override {
        ImGui::Begin("Volume Illumination");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size())))
            buildVolume(m_datasetIndex);
        ImGui::SliderInt("Slices", &m_numSlices, 16, 768);
        ImGui::SeparatorText("Light");
        ImGui::SliderFloat("Azimuth", &m_lightAzimuth, -3.14159f, 3.14159f);
        ImGui::SliderFloat("Elevation", &m_lightElevation, -1.5f, 1.5f);
        ImGui::SliderFloat("Ambient", &m_ambient, 0.0f, 1.0f);
        ImGui::SeparatorText("Material");
        ImGui::SliderFloat("Diffuse kd", &m_kd, 0.0f, 2.0f);
        ImGui::SliderFloat("Specular ks", &m_ks, 0.0f, 2.0f);
        ImGui::SliderFloat("Shininess", &m_shininess, 1.0f, 128.0f);
        ImGui::SliderFloat("Shading strength", &m_shadeStrength, 0.0f, 1.0f);
        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::End();
    }

    void onRender(gfx::Frame& frame) override {
        glm::vec3 viewDir = m_camera.forward();
        auto verts = volume::SliceProxyGeometry::generate(
            -m_boxHalf, m_boxHalf, viewDir, m_numSlices, /*reverse=*/true);
        uint32_t count = m_sliceBuffers->upload(frame.frameIndex, verts);
        if (count == 0) return;

        float aspect = static_cast<float>(swapchain().extent().width) /
                       static_cast<float>(swapchain().extent().height);

        glm::vec3 lightDir{std::cos(m_lightElevation) * std::sin(m_lightAzimuth),
                           std::sin(m_lightElevation),
                           std::cos(m_lightElevation) * std::cos(m_lightAzimuth)};

        PushConstants pc{};
        pc.mvp = m_camera.projection(aspect) * m_camera.view();
        pc.texelSize = glm::vec4(m_texelSize, 0.0f);
        pc.lightDir = glm::vec4(glm::normalize(lightDir), m_ambient);
        pc.viewDir = glm::vec4(viewDir, m_shininess);
        pc.params = glm::vec4(
            m_kd, m_ks, m_shadeStrength,
            static_cast<float>(kReferenceSlices) / static_cast<float>(m_numSlices));

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
    gfx::Pipeline m_pipeline;
    vk::raii::DescriptorSetLayout m_setLayout{nullptr};
    vk::raii::DescriptorPool m_descPool{nullptr};
    vk::raii::DescriptorSets m_descSets{nullptr};

    glm::vec3 m_boxHalf{0.5f};
    glm::vec3 m_texelSize{1.0f / 128};
    int m_datasetIndex = 0;
    int m_numSlices = 256;
    float m_lightAzimuth = 0.8f;
    float m_lightElevation = 0.6f;
    float m_ambient = 0.25f;
    float m_kd = 1.0f;
    float m_ks = 0.6f;
    float m_shininess = 32.0f;
    float m_shadeStrength = 1.0f;
};

int main() {
    try {
        gfx::setShaderDir(std::string("shaders/") + VVE_SHADER_SUBDIR);
        IlluminationApp app;
        app.run();
    } catch (const std::exception& e) {
        vve::log::error("Fatal: %s", e.what());
        return 1;
    }
    return 0;
}
