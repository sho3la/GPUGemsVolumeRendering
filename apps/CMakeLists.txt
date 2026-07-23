# ------------------------------------------------------------------------------
# Helper for declaring a chapter app: one executable linked against the engine,
# with its GLSL shaders compiled to SPIR-V next to the binary.
# ------------------------------------------------------------------------------
function(add_volume_app NAME)
    cmake_parse_arguments(APP "" "" "SOURCES;SHADERS" ${ARGN})

    add_executable(${NAME} ${APP_SOURCES})
    target_link_libraries(${NAME} PRIVATE vve::engine)
    set_target_properties(${NAME} PROPERTIES FOLDER "apps")

    if(APP_SHADERS)
        target_compile_shaders(${NAME} SHADERS ${APP_SHADERS})
    endif()
endfunction()

add_subdirectory(01_basic_emissive)
# Apps below are added as each is implemented.
add_subdirectory(02_transfer_functions)
add_subdirectory(03_illumination)
add_subdirectory(04_volumetric_shadows)
add_subdirectory(05_translucency)
add_subdirectory(06_procedural)
