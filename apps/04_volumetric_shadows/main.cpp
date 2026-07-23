// -----------------------------------------------------------------------------
// App 04 - Volumetric shadows via half-angle slicing (GPU Gems §39.5.1,
// Algorithm 39-3).
//
// Slices are taken perpendicular to the half-angle vector between the eye and
// the light so each slice can be rendered from both viewpoints. Two interleaved
// passes per slice:
//   * Light pass  - accumulates opacity into an offscreen light buffer; each
//                   slice attenuates the light reaching deeper slices.
//   * Eye pass    - modulates the classified color by the light still available
//                   at that slice (sampled from the light buffer) and composites
//                   it into an offscreen eye buffer.
// The eye buffer is finally composited to the screen.
//
// Note: this implements the dot(view, light) >= 0 branch of Algorithm 39-3 with
// a runtime "flip slice order" toggle to accommodate the opposite arrangement.
// Requires maxPushConstantsSize >= 132 (true on all modern desktop GPUs).
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
struct LightPC {
    glm::mat4 lightMVP;
    float opacityCorrection;
};
struct EyePC {
    glm::mat4 eyeMVP;
    glm::mat4 lightMVP;
    float opacityCorrection;
};
constexpr vk::Format kHDR = vk::Format::eR16G16B16A16Sfloat;
constexpr uint32_t kLightSize = 512;
constexpr int kReferenceSlices = 256;
constexpr size_t kMaxSliceVerts = 512 * 18;
} // namespace

class VolumetricShadowApp : public core::Application {
public:
    VolumetricShadowApp()
        : core::Application(makeConfig()), m_arcball(m_camera) {}

private:
    static Config makeConfig() {
        Config c;
        c.window.title = "GPU Gems 39 - 04 Volumetric Shadows";
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

        createLightBuffer();
        createEyeBuffer(swapchain().extent());
        createDescriptors();
        uploadTransferFunction(volume::TransferFunction::fire());
        buildVolume(m_datasetIndex);
        createPipelines();
    }

    void onResize(uint32_t, uint32_t) override {
        ctx().waitIdle();
        createEyeBuffer(swapchain().extent());
        writeCompositeDescriptor();
    }

    static gfx::Texture makeColorTarget(gfx::Context& ctx, vk::Extent2D ext,
                                        vk::Format fmt) {
        gfx::Texture::Desc d{};
        d.extent = vk::Extent3D{ext.width, ext.height, 1};
        d.format = fmt;
        d.type = vk::ImageType::e2D;
        d.usage = vk::ImageUsageFlagBits::eColorAttachment |
                  vk::ImageUsageFlagBits::eSampled;
        return gfx::Texture(ctx, d);
    }

    void createLightBuffer() {
        m_lightBuffer = makeColorTarget(ctx(), {kLightSize, kLightSize}, kHDR);
    }
    void createEyeBuffer(vk::Extent2D ext) {
        m_eyeExtent = ext;
        m_eyeBuffer = makeColorTarget(ctx(), ext, kHDR);
    }

    void createDescriptors() {
        auto sampler = [](uint32_t b) {
            return vk::DescriptorSetLayoutBinding{
                b, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment};
        };
        // volume + tf (light pass)
        std::array<vk::DescriptorSetLayoutBinding, 2> volB{sampler(0), sampler(1)};
        vk::DescriptorSetLayoutCreateInfo volInfo{};
        volInfo.setBindings(volB);
        m_volLayout = vk::raii::DescriptorSetLayout{ctx().device(), volInfo};

        // volume + tf + light buffer (eye pass)
        std::array<vk::DescriptorSetLayoutBinding, 3> eyeB{sampler(0), sampler(1),
                                                           sampler(2)};
        vk::DescriptorSetLayoutCreateInfo eyeInfo{};
        eyeInfo.setBindings(eyeB);
        m_eyeLayout = vk::raii::DescriptorSetLayout{ctx().device(), eyeInfo};

        // eye buffer (composite)
        std::array<vk::DescriptorSetLayoutBinding, 1> compB{sampler(0)};
        vk::DescriptorSetLayoutCreateInfo compInfo{};
        compInfo.setBindings(compB);
        m_compLayout = vk::raii::DescriptorSetLayout{ctx().device(), compInfo};

        std::array<vk::DescriptorPoolSize, 1> sizes{
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 8}};
        vk::DescriptorPoolCreateInfo poolInfo{
            vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, 3};
        poolInfo.setPoolSizes(sizes);
        m_descPool = vk::raii::DescriptorPool{ctx().device(), poolInfo};

        auto alloc = [&](vk::raii::DescriptorSetLayout& layout) {
            vk::DescriptorSetLayout raw = *layout;
            vk::DescriptorSetAllocateInfo ai{*m_descPool, 1, &raw};
            return std::move(vk::raii::DescriptorSets{ctx().device(), ai}.front());
        };
        m_volSet = alloc(m_volLayout);
        m_eyeSet = alloc(m_eyeLayout);
        m_compSet = alloc(m_compLayout);

        // Transfer-function texture.
        gfx::Texture::Desc d1{};
        d1.extent = vk::Extent3D{256, 1, 1};
        d1.format = vk::Format::eR8G8B8A8Unorm;
        d1.type = vk::ImageType::e1D;
        m_tfTex = gfx::Texture(ctx(), d1);

        // Static writes: light buffer into eye set, eye buffer into composite set.
        writeSampler(*m_eyeSet, 2, m_lightBuffer.view(), m_lightBuffer.sampler());
        writeCompositeDescriptor();
    }

    void writeSampler(vk::DescriptorSet set, uint32_t binding, vk::ImageView view,
                      vk::Sampler sampler) {
        vk::DescriptorImageInfo info{sampler, view,
                                     vk::ImageLayout::eShaderReadOnlyOptimal};
        vk::WriteDescriptorSet w{};
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorType = vk::DescriptorType::eCombinedImageSampler;
        w.setImageInfo(info);
        ctx().device().updateDescriptorSets(w, {});
    }

    void writeCompositeDescriptor() {
        writeSampler(*m_compSet, 0, m_eyeBuffer.view(), m_eyeBuffer.sampler());
    }

    void uploadTransferFunction(const volume::TransferFunction& tf) {
        auto data = tf.bakeRGBA8(256, /*premultiply=*/false);
        m_tfTex.uploadFromData(ctx(), data.data(), data.size());
        writeSampler(*m_volSet, 1, m_tfTex.view(), m_tfTex.sampler());
        writeSampler(*m_eyeSet, 1, m_tfTex.view(), m_tfTex.sampler());
    }

    void buildVolume(int index) {
        ctx().waitIdle();
        volume::VolumeData vol = m_source.create(index, 128);
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

        writeSampler(*m_volSet, 0, m_volumeTex.view(), m_volumeTex.sampler());
        writeSampler(*m_eyeSet, 0, m_volumeTex.view(), m_volumeTex.sampler());
    }

    void createPipelines() {
        gfx::ShaderModule fsVert(ctx().device(), "fullscreen.vert.spv");
        gfx::ShaderModule compFrag(ctx().device(), "composite.frag.spv");
        gfx::ShaderModule lightVert(ctx().device(), "light.vert.spv");
        gfx::ShaderModule lightFrag(ctx().device(), "light.frag.spv");
        gfx::ShaderModule eyeVert(ctx().device(), "eye.vert.spv");
        gfx::ShaderModule eyeFrag(ctx().device(), "eye.frag.spv");

        auto bindings = volume::SliceGeometryBuffers::bindings();
        auto attributes = volume::SliceGeometryBuffers::attributes();

        // Light pipeline.
        {
            vk::PushConstantRange pc{vk::ShaderStageFlagBits::eVertex |
                                         vk::ShaderStageFlagBits::eFragment,
                                     0, sizeof(LightPC)};
            gfx::GraphicsPipelineBuilder b;
            m_lightPipeline =
                b.setShaders(lightVert.stage(vk::ShaderStageFlagBits::eVertex),
                             lightFrag.stage(vk::ShaderStageFlagBits::eFragment))
                    .setVertexInput(bindings, attributes)
                    .setColorFormat(kHDR)
                    .setBlendAttachment(
                        gfx::GraphicsPipelineBuilder::blendPremultipliedOver())
                    .addDescriptorSetLayout(*m_volLayout)
                    .addPushConstantRange(pc)
                    .build(ctx().device());
        }
        // Eye pipeline (front-to-back "under").
        {
            vk::PushConstantRange pc{vk::ShaderStageFlagBits::eVertex |
                                         vk::ShaderStageFlagBits::eFragment,
                                     0, sizeof(EyePC)};
            gfx::GraphicsPipelineBuilder b;
            m_eyePipeline =
                b.setShaders(eyeVert.stage(vk::ShaderStageFlagBits::eVertex),
                             eyeFrag.stage(vk::ShaderStageFlagBits::eFragment))
                    .setVertexInput(bindings, attributes)
                    .setColorFormat(kHDR)
                    .setBlendAttachment(
                        gfx::GraphicsPipelineBuilder::blendUnder())
                    .addDescriptorSetLayout(*m_eyeLayout)
                    .addPushConstantRange(pc)
                    .build(ctx().device());
        }
        // Composite pipeline.
        {
            gfx::GraphicsPipelineBuilder b;
            m_compositePipeline =
                b.setShaders(fsVert.stage(vk::ShaderStageFlagBits::eVertex),
                             compFrag.stage(vk::ShaderStageFlagBits::eFragment))
                    .setVertexInput({}, {})
                    .setColorFormat(swapchain().format())
                    .setBlendAttachment(
                        gfx::GraphicsPipelineBuilder::blendPremultipliedOver())
                    .addDescriptorSetLayout(*m_compLayout)
                    .build(ctx().device());
        }
    }

    // Begins dynamic rendering into a color target; `flipY` matches the screen
    // orientation used by the swapchain pass.
    void beginColorPass(vk::raii::CommandBuffer& cmd, gfx::Texture& tex,
                        vk::Extent2D ext, bool clear,
                        const std::array<float, 4>& clearCol, bool flipY) {
        vk::RenderingAttachmentInfo att{};
        att.imageView = tex.view();
        att.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        att.loadOp = clear ? vk::AttachmentLoadOp::eClear
                           : vk::AttachmentLoadOp::eLoad;
        att.storeOp = vk::AttachmentStoreOp::eStore;
        att.clearValue.color = vk::ClearColorValue{clearCol[0], clearCol[1],
                                                   clearCol[2], clearCol[3]};
        vk::RenderingInfo ri{};
        ri.renderArea = vk::Rect2D{{0, 0}, ext};
        ri.layerCount = 1;
        ri.setColorAttachments(att);
        cmd.beginRendering(ri);

        vk::Viewport vp = flipY
            ? vk::Viewport{0, static_cast<float>(ext.height),
                           static_cast<float>(ext.width),
                           -static_cast<float>(ext.height), 0, 1}
            : vk::Viewport{0, 0, static_cast<float>(ext.width),
                           static_cast<float>(ext.height), 0, 1};
        cmd.setViewport(0, vp);
        cmd.setScissor(0, vk::Rect2D{{0, 0}, ext});
    }

    void onRecordOffscreen(gfx::Frame& frame) override {
        // Object-space directions (model = identity).
        glm::vec3 toEye = -m_camera.forward();
        glm::vec3 lightTravel{
            std::cos(m_lightElevation) * std::sin(m_lightAzimuth),
            std::sin(m_lightElevation),
            std::cos(m_lightElevation) * std::cos(m_lightAzimuth)};
        lightTravel = glm::normalize(lightTravel);
        glm::vec3 toLight = -lightTravel;
        glm::vec3 halfVec = glm::normalize(toEye + toLight);

        auto mesh = volume::SliceProxyGeometry::generateSliced(
            -m_boxHalf, m_boxHalf, halfVec, m_numSlices, m_flipOrder);
        m_lastSliceCount = static_cast<int>(mesh.sliceCount());
        if (mesh.vertices.empty()) return;
        m_sliceBuffers->upload(frame.frameIndex, mesh.vertices);
        vk::Buffer vb = m_sliceBuffers->buffer(frame.frameIndex);

        // Matrices.
        float aspect = static_cast<float>(m_eyeExtent.width) /
                       static_cast<float>(m_eyeExtent.height);
        glm::mat4 eyeMVP = m_camera.projection(aspect) * m_camera.view();

        glm::vec3 up = std::abs(toLight.y) < 0.99f ? glm::vec3(0, 1, 0)
                                                   : glm::vec3(1, 0, 0);
        glm::mat4 lightView = glm::lookAt(toLight * 1.5f, glm::vec3(0.0f), up);
        glm::mat4 lightProj = glm::ortho(-0.8f, 0.8f, -0.8f, 0.8f, 0.1f, 3.0f);
        glm::mat4 lightMVP = lightProj * lightView;

        float opacityCorrection =
            static_cast<float>(kReferenceSlices) / static_cast<float>(m_numSlices);

        auto& cmd = *frame.cmd;

        // Clear eye and light buffers.
        m_eyeBuffer.transition(cmd, vk::ImageLayout::eColorAttachmentOptimal,
                               vk::PipelineStageFlagBits2::eTopOfPipe,
                               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                               {}, vk::AccessFlagBits2::eColorAttachmentWrite);
        beginColorPass(cmd, m_eyeBuffer, m_eyeExtent, true, {0, 0, 0, 0}, true);
        cmd.endRendering();

        m_lightBuffer.transition(cmd, vk::ImageLayout::eColorAttachmentOptimal,
                                 vk::PipelineStageFlagBits2::eTopOfPipe,
                                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                 {}, vk::AccessFlagBits2::eColorAttachmentWrite);
        beginColorPass(cmd, m_lightBuffer, {kLightSize, kLightSize}, true,
                       {m_lightColor.r, m_lightColor.g, m_lightColor.b, 0.0f},
                       false);
        cmd.endRendering();
        m_lightBuffer.transition(cmd, vk::ImageLayout::eShaderReadOnlyOptimal,
                                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                 vk::PipelineStageFlagBits2::eFragmentShader,
                                 vk::AccessFlagBits2::eColorAttachmentWrite,
                                 vk::AccessFlagBits2::eShaderRead);

        LightPC lpc{lightMVP, opacityCorrection};
        EyePC epc{eyeMVP, lightMVP, opacityCorrection};
        vk::DeviceSize zero = 0;

        for (int i = 0; i < m_lastSliceCount; ++i) {
            uint32_t first = mesh.firstVertex[i];
            uint32_t count = mesh.vertexCount[i];

            // ---- Eye pass: sample the (attenuated) light buffer -------------
            m_eyeBuffer.transition(
                cmd, vk::ImageLayout::eColorAttachmentOptimal,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::AccessFlagBits2::eColorAttachmentWrite |
                    vk::AccessFlagBits2::eColorAttachmentRead);
            beginColorPass(cmd, m_eyeBuffer, m_eyeExtent, false, {0, 0, 0, 0}, true);
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_eyePipeline.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                   *m_eyePipeline.layout, 0, *m_eyeSet, {});
            cmd.pushConstants<EyePC>(*m_eyePipeline.layout,
                                     vk::ShaderStageFlagBits::eVertex |
                                         vk::ShaderStageFlagBits::eFragment,
                                     0, epc);
            cmd.bindVertexBuffers(0, vb, zero);
            cmd.draw(count, 1, first, 0);
            cmd.endRendering();

            // ---- Light pass: accumulate opacity into the light buffer -------
            m_lightBuffer.transition(
                cmd, vk::ImageLayout::eColorAttachmentOptimal,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::AccessFlagBits2::eShaderRead,
                vk::AccessFlagBits2::eColorAttachmentWrite);
            beginColorPass(cmd, m_lightBuffer, {kLightSize, kLightSize}, false,
                           {0, 0, 0, 0}, false);
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *m_lightPipeline.pipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                   *m_lightPipeline.layout, 0, *m_volSet, {});
            cmd.pushConstants<LightPC>(*m_lightPipeline.layout,
                                       vk::ShaderStageFlagBits::eVertex |
                                           vk::ShaderStageFlagBits::eFragment,
                                       0, lpc);
            cmd.bindVertexBuffers(0, vb, zero);
            cmd.draw(count, 1, first, 0);
            cmd.endRendering();

            m_lightBuffer.transition(
                cmd, vk::ImageLayout::eShaderReadOnlyOptimal,
                vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                vk::PipelineStageFlagBits2::eFragmentShader,
                vk::AccessFlagBits2::eColorAttachmentWrite,
                vk::AccessFlagBits2::eShaderRead);
        }

        // Eye buffer becomes a sampled texture for the composite pass.
        m_eyeBuffer.transition(cmd, vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits2::eFragmentShader,
                               vk::AccessFlagBits2::eColorAttachmentWrite,
                               vk::AccessFlagBits2::eShaderRead);
    }

    void onImGui() override {
        ImGui::Begin("Volumetric Shadows");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size())))
            buildVolume(m_datasetIndex);
        ImGui::SliderInt("Slices", &m_numSlices, 16, 256);
        ImGui::SeparatorText("Light");
        ImGui::SliderFloat("Azimuth", &m_lightAzimuth, -3.14159f, 3.14159f);
        ImGui::SliderFloat("Elevation", &m_lightElevation, -1.5f, 1.5f);
        ImGui::ColorEdit3("Light color", &m_lightColor.x);
        ImGui::Checkbox("Flip slice order", &m_flipOrder);
        ImGui::Text("Slices drawn: %d", m_lastSliceCount);
        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::End();
    }

    void onRender(gfx::Frame& frame) override {
        auto& cmd = *frame.cmd;
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,
                         *m_compositePipeline.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                               *m_compositePipeline.layout, 0, *m_compSet, {});
        cmd.draw(3, 1, 0, 0);
    }

    scene::Camera m_camera;
    scene::ArcballController m_arcball;
    volume::VolumeSource m_source;
    std::vector<const char*> m_datasetNames;
    std::unique_ptr<volume::SliceGeometryBuffers> m_sliceBuffers;

    gfx::Texture m_volumeTex;
    gfx::Texture m_tfTex;
    gfx::Texture m_lightBuffer;
    gfx::Texture m_eyeBuffer;
    vk::Extent2D m_eyeExtent{};

    gfx::Pipeline m_lightPipeline;
    gfx::Pipeline m_eyePipeline;
    gfx::Pipeline m_compositePipeline;

    vk::raii::DescriptorSetLayout m_volLayout{nullptr};
    vk::raii::DescriptorSetLayout m_eyeLayout{nullptr};
    vk::raii::DescriptorSetLayout m_compLayout{nullptr};
    vk::raii::DescriptorPool m_descPool{nullptr};
    vk::raii::DescriptorSet m_volSet{nullptr};
    vk::raii::DescriptorSet m_eyeSet{nullptr};
    vk::raii::DescriptorSet m_compSet{nullptr};

    glm::vec3 m_boxHalf{0.5f};
    int m_datasetIndex = 0;
    int m_numSlices = 128;
    int m_lastSliceCount = 0;
    float m_lightAzimuth = 1.0f;
    float m_lightElevation = 0.7f;
    glm::vec3 m_lightColor{1.0f, 0.95f, 0.85f};
    bool m_flipOrder = false;
};

int main() {
    try {
        gfx::setShaderDir(std::string("shaders/") + VVE_SHADER_SUBDIR);
        VolumetricShadowApp app;
        app.run();
    } catch (const std::exception& e) {
        vve::log::error("Fatal: %s", e.what());
        return 1;
    }
    return 0;
}
