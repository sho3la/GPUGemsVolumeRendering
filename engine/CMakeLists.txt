# ------------------------------------------------------------------------------
# VolumeEngine: reusable rendering engine shared by every chapter app.
# ------------------------------------------------------------------------------
add_library(VolumeEngine STATIC
    src/core/Window.cpp
    src/core/Application.cpp
    src/core/Paths.cpp
    src/gfx/vma_impl.cpp
    src/gfx/Context.cpp
    src/gfx/Swapchain.cpp
    src/gfx/Renderer.cpp
    src/gfx/Buffer.cpp
    src/gfx/Texture.cpp
    src/gfx/ShaderModule.cpp
    src/gfx/PipelineBuilder.cpp
    src/scene/Camera.cpp
    src/scene/ArcballController.cpp
    src/ui/ImGuiLayer.cpp
    src/volume/VolumeData.cpp
    src/volume/GradientComputer.cpp
    src/volume/SliceProxyGeometry.cpp
    src/volume/SliceGeometryBuffers.cpp
    src/volume/TransferFunction.cpp
    src/volume/VolumeSource.cpp)

add_library(vve::engine ALIAS VolumeEngine)

target_include_directories(VolumeEngine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(VolumeEngine PUBLIC
    Vulkan::Vulkan
    glfw
    glm::glm
    GPUOpen::VulkanMemoryAllocator
    imgui)

# Baked-in path to the repository data/ folder so in-tree dev builds find real
# datasets downloaded there without copying them next to the executable.
target_compile_definitions(VolumeEngine PRIVATE
    VVE_SOURCE_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

target_compile_definitions(VolumeEngine PUBLIC
    GLFW_INCLUDE_NONE
    GLM_FORCE_RADIANS
    GLM_FORCE_DEPTH_ZERO_TO_ONE          # Vulkan clip space
    GLM_ENABLE_EXPERIMENTAL)

# glm target name compatibility (older/newer glm export differ)
if(NOT TARGET glm::glm)
    target_link_libraries(VolumeEngine PUBLIC glm)
endif()
