cmake_minimum_required(VERSION 3.24)

project(GPUGemsVolumeRendering
    VERSION 1.0.0
    DESCRIPTION "Implementations of GPU Gems Ch.39 Volume Rendering Techniques"
    LANGUAGES CXX)

# ------------------------------------------------------------------------------
# Global settings
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "" FORCE)
endif()

# Put all runtime artifacts in one place so apps find their compiled shaders.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

if(MSVC)
    add_compile_options(/MP /permissive- /Zc:__cplusplus)
    # Vulkan-Hpp / GLM are heavy on template instantiations.
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
find_package(Vulkan REQUIRED)

include(Dependencies)   # cmake/Dependencies.cmake  (FetchContent)
include(CompileShaders) # cmake/CompileShaders.cmake

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------
add_subdirectory(engine)
add_subdirectory(apps)
