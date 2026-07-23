# ------------------------------------------------------------------------------
# Third-party dependencies pulled in via FetchContent.
# Vulkan itself comes from the installed SDK (find_package above).
# ------------------------------------------------------------------------------
include(FetchContent)

set(FETCHCONTENT_QUIET OFF)

# ---- GLFW --------------------------------------------------------------------
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(glfw
    GIT_REPOSITORY https://github.com/glfw/glfw.git
    GIT_TAG        3.4
    GIT_SHALLOW    TRUE)

# ---- GLM ---------------------------------------------------------------------
FetchContent_Declare(glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG        1.0.1
    GIT_SHALLOW    TRUE)

# ---- Vulkan Memory Allocator -------------------------------------------------
FetchContent_Declare(vma
    GIT_REPOSITORY https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator.git
    GIT_TAG        v3.1.0
    GIT_SHALLOW    TRUE)

# ---- Dear ImGui (no CMake of its own; we build it ourselves) -----------------
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG        v1.90.9
    GIT_SHALLOW    TRUE)

FetchContent_MakeAvailable(glfw glm vma imgui)

# ------------------------------------------------------------------------------
# Build Dear ImGui as a static library with GLFW + Vulkan backends.
# ------------------------------------------------------------------------------
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_vulkan.cpp)

target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${imgui_SOURCE_DIR}/backends)

target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)

# IMPORTANT: do NOT define IMGUI_IMPL_VULKAN_NO_PROTOTYPES here. ImGui's backend
# tests it with `#if defined(...)`, so defining it to *any* value (even 0) makes
# the backend expect function pointers loaded via ImGui_ImplVulkan_LoadFunctions()
# and leaves every Vulkan entry point null -> crash inside ImGui_ImplVulkan_Init.
# We link the SDK loader (Vulkan::Vulkan) and use the static prototypes directly.
