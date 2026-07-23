#include "vve/gfx/ShaderModule.hpp"

#include "vve/core/Paths.hpp"

#include <filesystem>
#include <fstream>
#include <vector>

namespace vve::gfx {

namespace {
std::vector<char> readFile(const std::string& path) {
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + path);
    }
    size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buffer(size);
    file.seekg(0);
    file.read(buffer.data(), static_cast<std::streamsize>(size));
    return buffer;
}
} // namespace

namespace {
std::string g_shaderDir = "shaders";
} // namespace

void setShaderDir(const std::string& dir) { g_shaderDir = dir; }

std::string shaderPath(const std::string& fileName) {
    // Prefer a location relative to the executable; fall back to the working
    // directory so running straight from build/bin also works.
    namespace fs = std::filesystem;
    fs::path candidates[] = {core::executableDir() / g_shaderDir / fileName,
                             fs::path(g_shaderDir) / fileName,
                             fs::current_path() / g_shaderDir / fileName};
    for (const auto& c : candidates) {
        if (fs::exists(c)) return c.string();
    }
    return (core::executableDir() / g_shaderDir / fileName).string();
}

ShaderModule::ShaderModule(vk::raii::Device& device,
                           const std::string& spvFileName) {
    auto code = readFile(shaderPath(spvFileName));
    vk::ShaderModuleCreateInfo info{};
    info.codeSize = code.size();
    info.pCode = reinterpret_cast<const uint32_t*>(code.data());
    m_module = vk::raii::ShaderModule{device, info};
}

vk::PipelineShaderStageCreateInfo ShaderModule::stage(
    vk::ShaderStageFlagBits stageFlags, const char* entry) const {
    vk::PipelineShaderStageCreateInfo info{};
    info.stage = stageFlags;
    info.module = *m_module;
    info.pName = entry;
    return info;
}

} // namespace vve::gfx
