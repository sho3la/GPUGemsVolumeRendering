add_volume_app(02_transfer_functions
    SOURCES main.cpp
    SHADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/slice.vert
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tf1d.frag
        ${CMAKE_CURRENT_SOURCE_DIR}/shaders/tf2d.frag)
