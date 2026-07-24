// -----------------------------------------------------------------------------
// App 05 - Translucency (GPU Gems §39.5.1).
//
// Extends the half-angle shadow algorithm (App 04) toward light that scatters
// through the medium. Two ideas from the chapter:
//   * Per-channel attenuation - the light buffer is attenuated with a color
//     absorption vector (multiplicative blend), so different wavelengths
//     penetrate to different depths (colored translucency / flashlight effect).
//   * Blur propagation - the eye pass gathers the light buffer through a blur
//     kernel (radius r = d*tan(phi), Equation 9), approximating the lateral
//     spread of forward-scattered light and softening the shadows.
//
// Implements the dot(view, light) >= 0 branch. Slices are always traversed
// front-to-back from the light (the only order the two-pass scheme needs), so
// there is no user-facing slice-order toggle.
// Requires maxPushConstantsSize >= 144 (true on all modern desktop GPUs).
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
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

using namespace vve;

namespace {
struct LightPC {
    glm::mat4 lightMVP;
    glm::vec4 absorption;
    float opacityCorrection;
};
struct EyePC {
    glm::mat4 eyeMVP;
    glm::mat4 lightMVP;
    glm::vec4 params; // x = opacityCorrection, y = blurRadius
};
// A transfer-function control point plus a human-readable label for the editor.
struct EditStop {
    std::string label;
    volume::TransferFunction::Stop s;
};
constexpr vk::Format kHDR = vk::Format::eR16G16B16A16Sfloat;
constexpr uint32_t kLightSize = 512;
constexpr int kReferenceSlices = 256;
constexpr size_t kMaxSliceVerts = 512 * 18;
} // namespace

class TranslucencyApp : public core::Application {
public:
    TranslucencyApp() : core::Application(makeConfig()), m_arcball(m_camera) {}

private:
    static Config makeConfig() {
        Config c;
        c.window.title = "GPU Gems 39 - 05 Translucency";
        return c;
    }

    void onInit() override {
        // The GPU Gems translucency figure sits on a black backdrop.
        clearColor = {0.0f, 0.0f, 0.0f, 1.0f};

        m_arcball.attach(window());
        m_arcball.setDistance(2.4f);
        m_sliceBuffers = std::make_unique<volume::SliceGeometryBuffers>(
            ctx(), gfx::Renderer::kFramesInFlight, kMaxSliceVerts);

        m_source.scanDirectory(core::dataDir());
        for (const auto& s : m_source.labels()) m_datasetNames.push_back(s.c_str());
        if (int r = m_source.firstRealIndex(); r >= 0) m_datasetIndex = r;
        // Default to the CT carp (the translucency scan from the chapter figure).
        for (int i = 0; i < static_cast<int>(m_datasetNames.size()); ++i) {
            if (std::string(m_datasetNames[i]).find("Carp") != std::string::npos) {
                m_datasetIndex = i;
                break;
            }
        }

        m_lightBuffer = makeColorTarget(ctx(), {kLightSize, kLightSize}, kHDR);
        createEyeBuffer(swapchain().extent());
        createDescriptors();
        applyTransferFunction();
        buildVolume(m_datasetIndex);
        createPipelines();
        frameDataset(m_datasetIndex);
    }

    // The carp lies along its longest axis (Z, the scan's slice stack); orbit
    // 90 deg around Y so we view it broadside, as in the figure.
    void frameDataset(int index) {
        bool elongated =
            index >= 0 && index < static_cast<int>(m_datasetNames.size()) &&
            std::string(m_datasetNames[index]).find("Carp") != std::string::npos;
        m_arcball.setOrientation(elongated ? 1.5708f : 0.0f, 0.08f);
    }

    // Named default transfer function for a dataset, keyed off its label. Each
    // real scan gets its own tissue bands (thresholds follow that scan's
    // histogram); anything unrecognised falls back to a generic fire ramp.
    static std::vector<EditStop> defaultStopsFor(const std::string& name) {
        auto has = [&](const char* s) { return name.find(s) != std::string::npos; };

        if (has("Carp")) {
            const glm::vec3 flesh(0.95f, 0.83f, 0.76f), bone(1.0f, 0.96f, 0.92f);
            return {
                {"Air",               {0.000f, glm::vec3(0.0f), 0.00f}},
                {"Flesh onset",       {0.045f, flesh, 0.00f}},
                {"Thin flesh / fins", {0.110f, flesh, 0.05f}},
                {"Muscle body",       {0.420f, flesh, 0.11f}},
                {"Skeleton",          {0.470f, bone, 0.40f}},
                {"Dense bone",        {0.640f, bone, 0.62f}},
                {"Densest bone",      {1.000f, glm::vec3(1.0f, 0.98f, 0.95f), 0.72f}},
            };
        }
        if (has("VisMale")) {
            // CT body: air, a big soft-tissue band (~64-79/255), then bone.
            const glm::vec3 skin(0.90f, 0.76f, 0.68f), bone(1.0f, 0.97f, 0.93f);
            return {
                {"Air",         {0.000f, glm::vec3(0.0f), 0.00f}},
                {"Skin",        {0.120f, skin, 0.03f}},
                {"Soft tissue", {0.250f, skin, 0.08f}},
                {"Muscle",      {0.330f, glm::vec3(0.88f, 0.72f, 0.66f), 0.11f}},
                {"Bone onset",  {0.400f, glm::vec3(0.95f, 0.92f, 0.88f), 0.40f}},
                {"Dense bone",  {0.650f, bone, 0.68f}},
                {"Densest bone",{1.000f, glm::vec3(1.0f, 0.99f, 0.96f), 0.78f}},
            };
        }
        if (has("CThead") || has("skull") || has("head")) {
            // Windowed CT head: air ~0, skin, then skull bone.
            const glm::vec3 skin(0.85f, 0.76f, 0.70f), bone(0.95f, 0.95f, 0.97f);
            return {
                {"Air",         {0.000f, glm::vec3(0.0f), 0.00f}},
                {"Skin",        {0.050f, skin, 0.03f}},
                {"Soft tissue", {0.280f, skin * 0.95f, 0.05f}},
                {"Bone onset",  {0.340f, glm::vec3(0.70f, 0.71f, 0.74f), 0.45f}},
                {"Dense bone",  {0.600f, bone, 0.80f}},
                {"Teeth / densest", {1.000f, glm::vec3(1.0f, 1.0f, 1.0f), 0.92f}},
            };
        }
        if (has("bonsai")) {
            const glm::vec3 leaf(0.05f, 0.65f, 0.08f), wood(0.55f, 0.30f, 0.05f);
            return {
                {"Air",           {0.000f, glm::vec3(0.0f), 0.00f}},
                {"Foliage onset", {0.137f, leaf, 0.00f}},
                {"Foliage",       {0.160f, leaf, 0.30f}},
                {"Foliage dense", {0.227f, leaf, 0.35f}},
                {"Wood",          {0.235f, wood, 0.55f}},
                {"Trunk / dense", {1.000f, glm::vec3(0.70f, 0.42f, 0.08f), 0.85f}},
            };
        }
        // Generic fallback: label the fire ramp's control points.
        std::vector<EditStop> out;
        auto stops = volume::TransferFunction::fire().stops();
        const char* names[] = {"Empty", "Low", "Mid", "High", "Peak"};
        for (size_t i = 0; i < stops.size(); ++i)
            out.push_back({i < 5 ? names[i] : "Stop", stops[i]});
        return out;
    }

    // Loads the current dataset's named default preset into the editable stop
    // list, then bakes. Labels make the UI editor self-explanatory.
    void applyTransferFunction() {
        std::string name =
            (m_datasetIndex >= 0 &&
             m_datasetIndex < static_cast<int>(m_datasetNames.size()))
                ? m_datasetNames[m_datasetIndex]
                : std::string();
        m_stops = defaultStopsFor(name);
        rebuildTransferFunction();
    }

    // Bakes the current (UI-editable) stop list into the transfer-function
    // texture. addStop sorts, so the baked ramp is correct regardless of the
    // order the stops appear in the editor.
    void rebuildTransferFunction() {
        volume::TransferFunction tf;
        for (const auto& e : m_stops) tf.addStop(e.s.t, e.s.color, e.s.opacity);
        uploadTransferFunction(tf);
    }

    void onResize(uint32_t, uint32_t) override {
        ctx().waitIdle();
        createEyeBuffer(swapchain().extent());
        writeSampler(*m_compSet, 0, m_eyeBuffer.view(), m_eyeBuffer.sampler());
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
        std::array<vk::DescriptorSetLayoutBinding, 2> volB{sampler(0), sampler(1)};
        vk::DescriptorSetLayoutCreateInfo volInfo{};
        volInfo.setBindings(volB);
        m_volLayout = vk::raii::DescriptorSetLayout{ctx().device(), volInfo};

        std::array<vk::DescriptorSetLayoutBinding, 3> eyeB{sampler(0), sampler(1),
                                                           sampler(2)};
        vk::DescriptorSetLayoutCreateInfo eyeInfo{};
        eyeInfo.setBindings(eyeB);
        m_eyeLayout = vk::raii::DescriptorSetLayout{ctx().device(), eyeInfo};

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

        gfx::Texture::Desc d1{};
        d1.extent = vk::Extent3D{256, 1, 1};
        d1.format = vk::Format::eR8G8B8A8Unorm;
        d1.type = vk::ImageType::e1D;
        m_tfTex = gfx::Texture(ctx(), d1);

        writeSampler(*m_eyeSet, 2, m_lightBuffer.view(), m_lightBuffer.sampler());
        writeSampler(*m_compSet, 0, m_eyeBuffer.view(), m_eyeBuffer.sampler());
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

    void uploadTransferFunction(const volume::TransferFunction& tf) {
        auto data = tf.bakeRGBA8(256, /*premultiply=*/false);
        m_tfTex.uploadFromData(ctx(), data.data(), data.size());
        writeSampler(*m_volSet, 1, m_tfTex.view(), m_tfTex.sampler());
        writeSampler(*m_eyeSet, 1, m_tfTex.view(), m_tfTex.sampler());
    }

    // Mirrors an R8 volume along Y (rows), leaving X and Z untouched.
    static std::vector<uint8_t> flipY(std::vector<uint8_t> src, int nx, int ny,
                                      int nz) {
        std::vector<uint8_t> dst(src.size());
        const size_t slice = static_cast<size_t>(nx) * ny;
        for (int z = 0; z < nz; ++z)
            for (int y = 0; y < ny; ++y) {
                const uint8_t* s = &src[z * slice + static_cast<size_t>(ny - 1 - y) * nx];
                std::copy(s, s + nx, &dst[z * slice + static_cast<size_t>(y) * nx]);
            }
        return dst;
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
        // Raw scans store rows top-to-bottom, so flip in Y to stand upright.
        auto bytes = flipY(vol.toR8(), vol.nx(), vol.ny(), vol.nz());
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
                        gfx::GraphicsPipelineBuilder::blendMultiply())
                    .addDescriptorSetLayout(*m_volLayout)
                    .addPushConstantRange(pc)
                    .build(ctx().device());
        }
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
        glm::vec3 toEye = -m_camera.forward();
        glm::vec3 lightTravel{
            std::cos(m_lightElevation) * std::sin(m_lightAzimuth),
            std::sin(m_lightElevation),
            std::cos(m_lightElevation) * std::cos(m_lightAzimuth)};
        lightTravel = glm::normalize(lightTravel);
        glm::vec3 toLight = -lightTravel;
        glm::vec3 halfVec = glm::normalize(toEye + toLight);

        // Slices are always iterated front-to-back from the light (the +halfVec
        // end): the half-vector lies between the eye and light, so that is the
        // only order the two-pass scheme needs. See App 04 for the full rationale.
        auto mesh = volume::SliceProxyGeometry::generateSliced(
            -m_boxHalf, m_boxHalf, halfVec, m_numSlices, /*reverse=*/true);
        m_lastSliceCount = static_cast<int>(mesh.sliceCount());
        if (mesh.vertices.empty()) return;
        m_sliceBuffers->upload(frame.frameIndex, mesh.vertices);
        vk::Buffer vb = m_sliceBuffers->buffer(frame.frameIndex);

        float aspect = static_cast<float>(m_eyeExtent.width) /
                       static_cast<float>(m_eyeExtent.height);
        glm::mat4 eyeMVP = m_camera.projection(aspect) * m_camera.view();
        glm::vec3 up = std::abs(toLight.y) < 0.99f ? glm::vec3(0, 1, 0)
                                                   : glm::vec3(1, 0, 0);
        glm::mat4 lightMVP = glm::ortho(-0.8f, 0.8f, -0.8f, 0.8f, 0.1f, 3.0f) *
                             glm::lookAt(toLight * 1.5f, glm::vec3(0.0f), up);

        float opacityCorrection =
            static_cast<float>(kReferenceSlices) / static_cast<float>(m_numSlices);

        auto& cmd = *frame.cmd;

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
                       {m_lightColor.r, m_lightColor.g, m_lightColor.b, 1.0f},
                       false);
        cmd.endRendering();
        m_lightBuffer.transition(cmd, vk::ImageLayout::eShaderReadOnlyOptimal,
                                 vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                                 vk::PipelineStageFlagBits2::eFragmentShader,
                                 vk::AccessFlagBits2::eColorAttachmentWrite,
                                 vk::AccessFlagBits2::eShaderRead);

        LightPC lpc{lightMVP, glm::vec4(m_absorption, 0.0f), opacityCorrection};
        EyePC epc{eyeMVP, lightMVP,
                  glm::vec4(opacityCorrection, m_blurRadius, 0.0f, 0.0f)};
        vk::DeviceSize zero = 0;

        for (int i = 0; i < m_lastSliceCount; ++i) {
            uint32_t first = mesh.firstVertex[i];
            uint32_t count = mesh.vertexCount[i];

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

        m_eyeBuffer.transition(cmd, vk::ImageLayout::eShaderReadOnlyOptimal,
                               vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                               vk::PipelineStageFlagBits2::eFragmentShader,
                               vk::AccessFlagBits2::eColorAttachmentWrite,
                               vk::AccessFlagBits2::eShaderRead);
    }

    void onImGui() override {
        ImGui::Begin("Translucency");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size()))) {
            buildVolume(m_datasetIndex);
            applyTransferFunction();
            frameDataset(m_datasetIndex);
        }
        ImGui::SliderInt("Slices", &m_numSlices, 16, 1024);
        ImGui::SeparatorText("Light");
        ImGui::SliderFloat("Azimuth", &m_lightAzimuth, -3.14159f, 3.14159f);
        ImGui::SliderFloat("Elevation", &m_lightElevation, -1.5f, 1.5f);
        ImGui::ColorEdit3("Light color", &m_lightColor.x);
        ImGui::SeparatorText("Scattering");
        ImGui::ColorEdit3("Absorption (per channel)", &m_absorption.x);
        ImGui::SliderFloat("Blur radius (phi)", &m_blurRadius, 0.0f, 0.02f,
                           "%.4f");

        // Live transfer-function editor. Each control point is a named tissue
        // band (Air / Flesh / Skeleton / ...) with its density position, colour
        // and opacity exposed, so the classification can be tuned in situ.
        ImGui::SeparatorText("Transfer function");
        bool tfChanged = false;
        int removeAt = -1;
        for (int i = 0; i < static_cast<int>(m_stops.size()); ++i) {
            ImGui::PushID(i);
            EditStop& e = m_stops[i];
            // Editable band name + its colour swatch, with a remove button.
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s", e.label.c_str());
            ImGui::ColorEdit3(
                "##col", &e.s.color.x,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            if (ImGui::IsItemEdited()) tfChanged = true;
            ImGui::SameLine();
            ImGui::PushItemWidth(150.0f);
            if (ImGui::InputText("##name", buf, sizeof(buf))) e.label = buf;
            ImGui::PopItemWidth();
            ImGui::SameLine();
            if (ImGui::SmallButton("remove")) removeAt = i;

            ImGui::PushItemWidth(220.0f);
            tfChanged |= ImGui::SliderFloat("density", &e.s.t, 0.0f, 1.0f, "%.3f");
            tfChanged |=
                ImGui::SliderFloat("opacity", &e.s.opacity, 0.0f, 1.0f, "%.2f");
            ImGui::PopItemWidth();
            ImGui::Spacing();
            ImGui::PopID();
        }
        if (removeAt >= 0 && m_stops.size() > 1) {
            m_stops.erase(m_stops.begin() + removeAt);
            tfChanged = true;
        }
        if (ImGui::SmallButton("+ Add band")) {
            m_stops.push_back({"New band", {0.5f, glm::vec3(1.0f), 0.5f}});
            tfChanged = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset preset")) applyTransferFunction();
        if (tfChanged) rebuildTransferFunction();

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

    std::vector<EditStop> m_stops; // UI-editable, labelled transfer function
    glm::vec3 m_boxHalf{0.5f};
    int m_datasetIndex = 0;
    int m_numSlices = 256;
    int m_lastSliceCount = 0;
    float m_lightAzimuth = 1.0f;
    float m_lightElevation = 0.7f;
    glm::vec3 m_lightColor{1.0f, 1.0f, 1.0f};
    // Strongly wavelength-dependent so deep flesh glows red (red penetrates,
    // green/blue are absorbed) - the translucency look of the chapter figure.
    glm::vec3 m_absorption{0.22f, 0.85f, 0.96f};
    float m_blurRadius = 0.006f;
};

int main() {
    try {
        gfx::setShaderDir(std::string("shaders/") + VVE_SHADER_SUBDIR);
        TranslucencyApp app;
        app.run();
    } catch (const std::exception& e) {
        vve::log::error("Fatal: %s", e.what());
        return 1;
    }
    return 0;
}
