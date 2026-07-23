#pragma once

// Central Vulkan configuration header. Include this instead of <vulkan/vulkan.hpp>
// so every translation unit sees the same macro configuration.

#include <vulkan/vulkan_raii.hpp>

#include <cstdint>
#include <stdexcept>
#include <string>

namespace vve::gfx {

// Throws std::runtime_error when a vk::Result indicates failure.
inline void check(vk::Result r, const char* what) {
    if (r != vk::Result::eSuccess) {
        throw std::runtime_error(std::string(what) + ": " +
                                 vk::to_string(r));
    }
}

inline void check(VkResult r, const char* what) {
    check(static_cast<vk::Result>(r), what);
}

} // namespace vve::gfx
