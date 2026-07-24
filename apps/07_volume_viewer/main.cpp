// -----------------------------------------------------------------------------
// App 07 - Versatile volume viewer.
//
// A general-purpose direct-volume-rendering front end built on the same engine
// as the chapter apps. It ties together the techniques the earlier apps explore
// individually so any dataset in data/ can be inspected quickly:
//   * Render modes: DVR (classified, optionally Blinn-Phong shaded) and MIP.
//   * Radiology-style window/level (WL/WW) plus a density clip window.
//   * A live, labelled transfer-function editor with per-dataset presets.
//   * Adjustable light, material and background.
// One fragment shader handles both modes; DVR binds a premultiplied-"over"
// pipeline (back-to-front slices) and MIP binds a max-blend pipeline.
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include <glm/gtc/matrix_transform.hpp>
#include <imgui.h>

using namespace vve;

namespace {
struct PushConstants {
    glm::mat4 mvp;
    glm::vec4 texelSize; // xyz = 1/dim, w = mode (0 = DVR, 1 = MIP)
    glm::vec4 lightDir;  // xyz dir, w = ambient
    glm::vec4 viewDir;   // xyz dir, w = shininess
    glm::vec4 params;    // x=kd, y=ks, z=shadeStrength, w=opacityCorrection
    glm::vec4 window;    // x=win low, y=win high, z=clip low, w=clip high
};

struct EditStop {
    std::string label;
    volume::TransferFunction::Stop s;
};

constexpr int kReferenceSlices = 256;
constexpr size_t kMaxSliceVerts = 1024 * 18;
const char* kModeNames[] = {"DVR (shaded)", "MIP"};
const char* kPresetNames[] = {"Auto", "Grayscale", "Fire", "Cool / Warm",
                              "CT Bone"};
} // namespace

class VolumeViewerApp : public core::Application {
public:
    VolumeViewerApp() : core::Application(makeConfig()), m_arcball(m_camera) {}

private:
    static Config makeConfig() {
        Config c;
        c.window.title = "GPU Gems 39 - 07 Volume Viewer";
        return c;
    }

    void onInit() override {
        clearColor = {0.0f, 0.0f, 0.0f, 1.0f};
        m_arcball.attach(window());
        m_arcball.setDistance(2.4f);

        m_sliceBuffers = std::make_unique<volume::SliceGeometryBuffers>(
            ctx(), gfx::Renderer::kFramesInFlight, kMaxSliceVerts);

        m_source.scanDirectory(core::dataDir());
        for (const auto& s : m_source.labels()) m_datasetNames.push_back(s.c_str());
        if (int r = m_source.firstRealIndex(); r >= 0) m_datasetIndex = r;
        for (int i = 0; i < static_cast<int>(m_datasetNames.size()); ++i) {
            if (std::string(m_datasetNames[i]).find("Lobster") != std::string::npos) {
                m_datasetIndex = i;
                break;
            }
        }

        createDescriptors();
        applyPreset(m_preset);
        buildVolume(m_datasetIndex);
        createPipelines();
    }

    // --- Transfer-function presets ------------------------------------------
    static std::vector<EditStop> labelled(const volume::TransferFunction& tf) {
        std::vector<EditStop> v;
        int i = 0;
        for (const auto& s : tf.stops())
            v.push_back({"Stop " + std::to_string(i++), s});
        return v;
    }

    std::vector<EditStop> presetStops(int preset) const {
        std::string name =
            (m_datasetIndex >= 0 &&
             m_datasetIndex < static_cast<int>(m_datasetNames.size()))
                ? m_datasetNames[m_datasetIndex]
                : std::string();
        auto has = [&](const char* s) { return name.find(s) != std::string::npos; };

        switch (preset) {
            case 1: return labelled(volume::TransferFunction::grayscaleRamp());
            case 2: return labelled(volume::TransferFunction::fire());
            case 3: return labelled(volume::TransferFunction::coolWarm());
            case 4: return labelled(volume::TransferFunction::ctSkull());
            case 0:
            default:
                if (has("Lobster")) {
                    // Air (74%) and a low-density mounting medium (~16-24/255)
                    // dominate; start the shell above it so the animal reads
                    // cleanly without a surrounding haze.
                    return {
                        {"Air / medium", {0.00f, glm::vec3(0.0f), 0.00f}},
                        {"Shell onset",  {0.105f, glm::vec3(0.45f, 0.40f, 0.95f), 0.00f}},
                        {"Shell",        {0.140f, glm::vec3(0.50f, 0.45f, 1.00f), 0.10f}},
                        {"Flesh",        {0.240f, glm::vec3(1.00f, 0.35f, 0.55f), 0.24f}},
                        {"Dense",        {0.450f, glm::vec3(1.00f, 0.55f, 0.45f), 0.42f}},
                        {"Densest",      {1.000f, glm::vec3(1.00f, 0.85f, 0.75f), 0.62f}},
                    };
                }
                if (has("Subclavia") || has("Angio")) {
                    // CT angiography: faint soft tissue, bright contrast-filled
                    // vessels, opaque calcium/bone.
                    return {
                        {"Air",          {0.00f, glm::vec3(0.0f), 0.000f}},
                        {"Soft tissue",  {0.10f, glm::vec3(0.50f, 0.12f, 0.10f), 0.015f}},
                        {"Vessels",      {0.20f, glm::vec3(0.95f, 0.20f, 0.15f), 0.35f}},
                        {"Dense / bone", {0.45f, glm::vec3(1.00f, 0.92f, 0.85f), 0.70f}},
                        {"Densest",      {1.00f, glm::vec3(1.00f, 0.97f, 0.92f), 0.85f}},
                    };
                }
                if (has("Carp")) return labelled(volume::TransferFunction::carp());
                if (has("bonsai")) return labelled(volume::TransferFunction::bonsai());
                if (has("CThead") || has("VisMale") || has("head"))
                    return labelled(volume::TransferFunction::ctSkull());
                return labelled(volume::TransferFunction::fire());
        }
    }

    void applyPreset(int preset) {
        m_preset = preset;
        m_stops = presetStops(preset);
        rebuildTransferFunction();
    }

    void rebuildTransferFunction() {
        volume::TransferFunction tf;
        for (const auto& e : m_stops) tf.addStop(e.s.t, e.s.color, e.s.opacity);
        auto data = tf.bakeRGBA8(256, /*premultiply=*/false);
        m_tfTex.uploadFromData(ctx(), data.data(), data.size());
        writeSampler(1, m_tfTex.view(), m_tfTex.sampler());
    }

    // --- Resources -----------------------------------------------------------
    void createDescriptors() {
        auto sampler = [](uint32_t b) {
            return vk::DescriptorSetLayoutBinding{
                b, vk::DescriptorType::eCombinedImageSampler, 1,
                vk::ShaderStageFlagBits::eFragment};
        };
        std::array<vk::DescriptorSetLayoutBinding, 2> bindings{sampler(0),
                                                               sampler(1)};
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

    // Mirrors an R8 volume along Y so top-to-bottom scans stand upright.
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
        m_texelSize = glm::vec3(1.0f / vol.nx(), 1.0f / vol.ny(), 1.0f / vol.nz());

        gfx::Texture::Desc desc{};
        desc.extent = vk::Extent3D{static_cast<uint32_t>(vol.nx()),
                                   static_cast<uint32_t>(vol.ny()),
                                   static_cast<uint32_t>(vol.nz())};
        desc.format = vk::Format::eR8Unorm;
        desc.type = vk::ImageType::e3D;
        m_volumeTex = gfx::Texture(ctx(), desc);
        auto bytes = flipY(vol.toR8(), vol.nx(), vol.ny(), vol.nz());
        m_volumeTex.uploadFromData(ctx(), bytes.data(), bytes.size());
        writeSampler(0, m_volumeTex.view(), m_volumeTex.sampler());
    }

    void createPipelines() {
        gfx::ShaderModule vert(ctx().device(), "vol.vert.spv");
        gfx::ShaderModule frag(ctx().device(), "vol.frag.spv");
        auto bindings = volume::SliceGeometryBuffers::bindings();
        auto attributes = volume::SliceGeometryBuffers::attributes();
        vk::PushConstantRange pcRange{
            vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment,
            0, sizeof(PushConstants)};

        auto buildWith = [&](const vk::PipelineColorBlendAttachmentState& blend) {
            gfx::GraphicsPipelineBuilder b;
            return b.setShaders(vert.stage(vk::ShaderStageFlagBits::eVertex),
                                frag.stage(vk::ShaderStageFlagBits::eFragment))
                .setVertexInput(bindings, attributes)
                .setColorFormat(swapchain().format())
                .setBlendAttachment(blend)
                .addDescriptorSetLayout(*m_setLayout)
                .addPushConstantRange(pcRange)
                .build(ctx().device());
        };
        m_dvrPipeline = buildWith(
            gfx::GraphicsPipelineBuilder::blendPremultipliedOver());
        m_mipPipeline = buildWith(gfx::GraphicsPipelineBuilder::blendMax());
    }

    void onImGui() override {
        ImGui::Begin("Volume Viewer");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size()))) {
            buildVolume(m_datasetIndex);
            applyPreset(m_preset);
        }
        ImGui::Combo("Render mode", &m_mode, kModeNames, IM_ARRAYSIZE(kModeNames));
        ImGui::SliderInt("Slices", &m_numSlices, 16, 1024);
        ImGui::ColorEdit3("Background", &clearColor[0]);

        ImGui::SeparatorText("Windowing (WL / WW)");
        ImGui::SliderFloat("Window level", &m_winLevel, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Window width", &m_winWidth, 0.02f, 1.0f, "%.3f");
        ImGui::DragFloatRange2("Clip range", &m_clipLo, &m_clipHi, 0.005f, 0.0f,
                               1.0f, "lo %.2f", "hi %.2f");

        if (m_mode == 0) {
            ImGui::SeparatorText("Transfer function");
            if (ImGui::Combo("Preset", &m_preset, kPresetNames,
                             IM_ARRAYSIZE(kPresetNames)))
                applyPreset(m_preset);
            drawTransferFunctionEditor();

            ImGui::SeparatorText("Light & material");
            ImGui::SliderFloat("Azimuth", &m_lightAzimuth, -3.14159f, 3.14159f);
            ImGui::SliderFloat("Elevation", &m_lightElevation, -1.5f, 1.5f);
            ImGui::SliderFloat("Ambient", &m_ambient, 0.0f, 1.0f);
            ImGui::SliderFloat("Diffuse kd", &m_kd, 0.0f, 2.0f);
            ImGui::SliderFloat("Specular ks", &m_ks, 0.0f, 2.0f);
            ImGui::SliderFloat("Shininess", &m_shininess, 1.0f, 128.0f);
            ImGui::SliderFloat("Shading strength", &m_shadeStrength, 0.0f, 1.0f);
        }

        ImGui::Text("%.1f FPS", ImGui::GetIO().Framerate);
        ImGui::End();
    }

    void drawTransferFunctionEditor() {
        bool changed = false;
        int removeAt = -1;
        for (int i = 0; i < static_cast<int>(m_stops.size()); ++i) {
            ImGui::PushID(i);
            EditStop& e = m_stops[i];
            char buf[64];
            std::snprintf(buf, sizeof(buf), "%s", e.label.c_str());
            ImGui::ColorEdit3(
                "##col", &e.s.color.x,
                ImGuiColorEditFlags_NoInputs | ImGuiColorEditFlags_NoLabel);
            if (ImGui::IsItemEdited()) changed = true;
            ImGui::SameLine();
            ImGui::PushItemWidth(150.0f);
            if (ImGui::InputText("##name", buf, sizeof(buf))) e.label = buf;
            ImGui::PopItemWidth();
            ImGui::SameLine();
            if (ImGui::SmallButton("remove")) removeAt = i;
            ImGui::PushItemWidth(220.0f);
            changed |= ImGui::SliderFloat("density", &e.s.t, 0.0f, 1.0f, "%.3f");
            changed |= ImGui::SliderFloat("opacity", &e.s.opacity, 0.0f, 1.0f, "%.2f");
            ImGui::PopItemWidth();
            ImGui::Spacing();
            ImGui::PopID();
        }
        if (removeAt >= 0 && m_stops.size() > 1) {
            m_stops.erase(m_stops.begin() + removeAt);
            changed = true;
        }
        if (ImGui::SmallButton("+ Add band")) {
            m_stops.push_back({"New band", {0.5f, glm::vec3(1.0f), 0.5f}});
            changed = true;
        }
        ImGui::SameLine();
        if (ImGui::SmallButton("Reset preset")) applyPreset(m_preset);
        if (changed) rebuildTransferFunction();
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
        pc.texelSize = glm::vec4(m_texelSize, static_cast<float>(m_mode));
        pc.lightDir = glm::vec4(glm::normalize(lightDir), m_ambient);
        pc.viewDir = glm::vec4(viewDir, m_shininess);
        pc.params = glm::vec4(
            m_kd, m_ks, m_shadeStrength,
            static_cast<float>(kReferenceSlices) / static_cast<float>(m_numSlices));
        float lo = m_winLevel - 0.5f * m_winWidth;
        float hi = m_winLevel + 0.5f * m_winWidth;
        pc.window = glm::vec4(lo, hi, m_clipLo, m_clipHi);

        auto& cmd = *frame.cmd;
        gfx::Pipeline& pipe = m_mode == 1 ? m_mipPipeline : m_dvrPipeline;
        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, *pipe.pipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, *pipe.layout, 0,
                               *m_descSets[0], {});
        cmd.pushConstants<PushConstants>(
            *pipe.layout,
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
    gfx::Pipeline m_dvrPipeline;
    gfx::Pipeline m_mipPipeline;
    vk::raii::DescriptorSetLayout m_setLayout{nullptr};
    vk::raii::DescriptorPool m_descPool{nullptr};
    vk::raii::DescriptorSets m_descSets{nullptr};

    std::vector<EditStop> m_stops;
    glm::vec3 m_boxHalf{0.5f};
    glm::vec3 m_texelSize{1.0f / 128};
    int m_datasetIndex = 0;
    int m_preset = 0;
    int m_mode = 0;
    int m_numSlices = 256;
    float m_winLevel = 0.5f;
    float m_winWidth = 1.0f;
    float m_clipLo = 0.0f;
    float m_clipHi = 1.0f;
    float m_lightAzimuth = 0.8f;
    float m_lightElevation = 0.6f;
    float m_ambient = 0.30f;
    float m_kd = 1.0f;
    float m_ks = 0.5f;
    float m_shininess = 32.0f;
    float m_shadeStrength = 1.0f;
};

int main() {
    try {
        gfx::setShaderDir(std::string("shaders/") + VVE_SHADER_SUBDIR);
        VolumeViewerApp app;
        app.run();
    } catch (const std::exception& e) {
        vve::log::error("Fatal: %s", e.what());
        return 1;
    }
    return 0;
}
