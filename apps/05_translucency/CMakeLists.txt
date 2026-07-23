add_volume_app(05_translucency
    SOURCES main.cpp
    SHADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/fullscreen.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/composite.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/light.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/light.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/eye.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/eye.frag)
