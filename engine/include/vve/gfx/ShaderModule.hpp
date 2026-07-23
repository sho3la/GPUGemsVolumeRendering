#pragma once

#include "vve/gfx/VulkanCommon.hpp"

#include <string>

namespace vve::gfx {

// Loads a compiled SPIR-V module from disk. Shaders are compiled by CMake into
// "<binary>/shaders/<name>.spv"; pass just the file name.
class ShaderModule {
public:
    ShaderModule(vk::raii::Device& device, const std::string& spvFileName);

    [[nodiscard]] vk::ShaderModule handle() const { return *m_module; }

    // Convenience for filling a pipeline stage.
    [[nodiscard]] vk::PipelineShaderStageCreateInfo stage(
        vk::ShaderStageFlagBits stageFlags,
        const char* entry = "main") const;

private:
    vk::raii::ShaderModule m_module{nullptr};
};

// Sets the directory (relative to the working dir) searched for compiled
// shaders. Each app calls this with its own "shaders/<app>" subdirectory so
// identically named shaders across apps don't collide.
void setShaderDir(const std::string& dir);

// Resolves a shader file name to an absolute path next to the executable.
std::string shaderPath(const std::string& fileName);

} // namespace vve::gfx
