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
    glm::vec4 texelSize;
    glm::vec4 lightDir;     // xyz dir, w = ambient
    glm::vec4 viewDir;      // xyz dir, w = shininess
    glm::vec4 params;       // x=kd, y=ks, z=shadeStrength, w=opacityCorrection
    glm::vec4 window;       // x=window low, y=window high (radiology WL/WW remap)
};

// One transfer-function control point plus a tissue label for the editor.
struct EditStop {
    std::string label;
    volume::TransferFunction::Stop s;
};

// Radiology-style volume-rendering presets (generic CT, defined over the
// scan's normalised value with air near 0). Each carries a default window.
struct VRPreset {
    const char* name;
    float level, width; // default window level / width
};
// The stops below are authored directly in the scan's normalised value, so each
// preset's default window is the identity [0,1] (level 0.5, width 1.0); WL/WW
// then acts as a live contrast tweak on top, like a CT console.
constexpr VRPreset kPresets[] = {
    {"CT Bone", 0.5f, 1.0f},
    {"CT Soft Tissue", 0.5f, 1.0f},
    {"CT Skin (surface)", 0.5f, 1.0f},
    {"CT Angio (vessels)", 0.5f, 1.0f},
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
        // Default to the VisMale CT body for the radiology-style presets.
        for (int i = 0; i < static_cast<int>(m_datasetNames.size()); ++i) {
            if (std::string(m_datasetNames[i]).find("VisMale") != std::string::npos) {
                m_datasetIndex = i;
                break;
            }
        }

        createDescriptors();
        applyPreset(m_preset);
        buildVolume(m_datasetIndex);
        createPipeline();
    }

    // Radiology CT VR presets: named tissue bands classified over the scan's
    // normalised value (air ~0). Loading a preset also sets its default window.
    void applyPreset(int preset) {
        m_preset = preset;
        m_winLevel = kPresets[preset].level;
        m_winWidth = kPresets[preset].width;
        m_stops.clear();
        switch (preset) {
            case 1: { // Soft tissue (+ bone)
                m_stops = {
                    {"Air",         {0.00f, glm::vec3(0.0f), 0.00f}},
                    {"Skin",        {0.10f, glm::vec3(0.85f, 0.62f, 0.50f), 0.02f}},
                    {"Fat",         {0.20f, glm::vec3(0.90f, 0.75f, 0.55f), 0.08f}},
                    {"Muscle",      {0.30f, glm::vec3(0.80f, 0.35f, 0.32f), 0.18f}},
                    {"Bone",        {0.43f, glm::vec3(0.95f, 0.92f, 0.86f), 0.60f}},
                    {"Dense bone",  {1.00f, glm::vec3(1.0f, 0.98f, 0.95f), 0.85f}},
                };
                break;
            }
            case 2: { // Skin / surface
                const glm::vec3 skin(0.88f, 0.68f, 0.56f);
                m_stops = {
                    {"Air",         {0.00f, glm::vec3(0.0f), 0.00f}},
                    {"Skin onset",  {0.09f, skin, 0.00f}},
                    {"Skin",        {0.13f, skin, 0.85f}},
                    {"Interior",    {0.50f, skin * 0.9f, 0.90f}},
                    {"Deep",        {1.00f, skin * 0.8f, 0.95f}},
                };
                break;
            }
            case 3: { // Angio / vessels
                m_stops = {
                    {"Air",         {0.00f, glm::vec3(0.0f), 0.00f}},
                    {"Soft tissue", {0.28f, glm::vec3(0.45f, 0.06f, 0.05f), 0.04f}},
                    {"Vessels",     {0.42f, glm::vec3(0.95f, 0.25f, 0.18f), 0.45f}},
                    {"Bone",        {0.62f, glm::vec3(1.0f, 0.95f, 0.90f), 0.80f}},
                    {"Dense bone",  {1.00f, glm::vec3(1.0f, 0.98f, 0.95f), 0.90f}},
                };
                break;
            }
            default: { // 0: Bone
                m_stops = {
                    {"Air",         {0.00f, glm::vec3(0.0f), 0.00f}},
                    {"Pre-bone",    {0.38f, glm::vec3(0.90f, 0.87f, 0.80f), 0.00f}},
                    {"Bone onset",  {0.43f, glm::vec3(0.92f, 0.88f, 0.80f), 0.55f}},
                    {"Dense bone",  {0.70f, glm::vec3(1.0f, 0.98f, 0.94f), 0.90f}},
                    {"Densest",     {1.00f, glm::vec3(1.0f, 1.0f, 0.98f), 1.00f}},
                };
                break;
            }
        }
        rebuildTransferFunction();
    }

    // Bakes the editable stop list into the transfer-function texture. addStop
    // sorts, so the ramp is correct regardless of the editor's row order.
    void rebuildTransferFunction() {
        volume::TransferFunction tf;
        for (const auto& e : m_stops) tf.addStop(e.s.t, e.s.color, e.s.opacity);
        uploadTransferFunction(tf);
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
        ImGui::Begin("CT Volume Rendering");
        if (ImGui::Combo("Dataset", &m_datasetIndex, m_datasetNames.data(),
                         static_cast<int>(m_datasetNames.size())))
            buildVolume(m_datasetIndex);

        // Radiology-style VR preset picker.
        ImGui::SeparatorText("VR preset");
        const char* presetNames[IM_ARRAYSIZE(kPresets)];
        for (int i = 0; i < IM_ARRAYSIZE(kPresets); ++i)
            presetNames[i] = kPresets[i].name;
        if (ImGui::Combo("Preset", &m_preset, presetNames,
                         IM_ARRAYSIZE(kPresets)))
            applyPreset(m_preset);

        // Window/level - the core radiology contrast control (WL/WW). Here in
        // normalised units [0,1]; narrowing the width increases contrast.
        ImGui::SeparatorText("Windowing (WL / WW)");
        ImGui::SliderFloat("Window level", &m_winLevel, 0.0f, 1.0f, "%.3f");
        ImGui::SliderFloat("Window width", &m_winWidth, 0.02f, 1.0f, "%.3f");
        ImGui::SliderInt("Slices", &m_numSlices, 16, 768);

        // Live, labelled transfer-function editor (opacity/colour curve).
        ImGui::SeparatorText("Transfer function");
        bool tfChanged = false;
        int removeAt = -1;
        for (int i = 0; i < static_cast<int>(m_stops.size()); ++i) {
            ImGui::PushID(i);
            EditStop& e = m_stops[i];
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
        if (ImGui::SmallButton("Reset preset")) applyPreset(m_preset);
        if (tfChanged) rebuildTransferFunction();

        ImGui::SeparatorText("Light");
        ImGui::SliderFloat("Azimuth", &m_lightAzimuth, -3.14159f, 3.14159f);
        ImGui::SliderFloat("Elevation", &m_lightElevation, -1.5f, 1.5f);
        ImGui::SliderFloat("Ambient", &m_ambient, 0.0f, 1.0f);
        ImGui::SeparatorText("Material (Blinn-Phong)");
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
        // Radiology windowing: map [level-width/2, level+width/2] onto the
        // transfer-function domain before classification.
        float lo = m_winLevel - 0.5f * m_winWidth;
        float hi = m_winLevel + 0.5f * m_winWidth;
        pc.window = glm::vec4(lo, hi, 0.0f, 0.0f);
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

    std::vector<EditStop> m_stops; // UI-editable, labelled transfer function
    glm::vec3 m_boxHalf{0.5f};
    glm::vec3 m_texelSize{1.0f / 128};
    int m_datasetIndex = 0;
    int m_preset = 0;
    int m_numSlices = 256;
    float m_winLevel = 0.5f; // radiology window level
    float m_winWidth = 1.0f; // radiology window width
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
