# ------------------------------------------------------------------------------
# target_compile_shaders(<target> SHADERS a.vert b.frag ...)
#
# Compiles each GLSL shader to SPIR-V with glslc and copies the results next to
# the runtime output as "<bin>/shaders/<name>.spv". The generated files are
# attached to <target> so they rebuild when the source changes.
# ------------------------------------------------------------------------------
find_program(GLSLC_EXECUTABLE
    NAMES glslc
    HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin" ${Vulkan_GLSLC_EXECUTABLE})

if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found. Ensure the Vulkan SDK is installed and VULKAN_SDK is set.")
endif()

function(target_compile_shaders TARGET)
    cmake_parse_arguments(ARG "" "" "SHADERS" ${ARGN})

    set(_spv_outputs "")
    # Namespace by target so equally named shaders in different apps don't clash.
    set(_out_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/shaders/${TARGET}")
    file(MAKE_DIRECTORY "${_out_dir}")
    target_compile_definitions(${TARGET} PRIVATE VVE_SHADER_SUBDIR="${TARGET}")

    foreach(_src ${ARG_SHADERS})
        get_filename_component(_name "${_src}" NAME)
        set(_out "${_out_dir}/${_name}.spv")
        add_custom_command(
            OUTPUT  "${_out}"
            COMMAND ${GLSLC_EXECUTABLE} --target-env=vulkan1.3 -O -o "${_out}" "${_src}"
            DEPENDS "${_src}"
            COMMENT "Compiling shader ${_name}"
            VERBATIM)
        list(APPEND _spv_outputs "${_out}")
    endforeach()

    add_custom_target(${TARGET}_shaders DEPENDS ${_spv_outputs})
    add_dependencies(${TARGET} ${TARGET}_shaders)
endfunction()
