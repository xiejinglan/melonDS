find_program(ELF2NRO elf2nro)
find_program(NACPTOOL nacptool)

project(switch)

SET(SOURCES_SWITCH
    main.cpp
    Platform.cpp
    profiler.cpp

    imgui/imgui_draw.cpp
    imgui/imgui_widgets.cpp
    imgui/imgui.cpp
    imgui/imgui_impl_opengl3.cpp
    imgui/imgui_demo.cpp
)

add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

add_executable(melonDS.elf ${SOURCES_SWITCH})
target_link_libraries(melonDS.elf core glad EGL glapi drm_nouveau nx)

add_custom_command(OUTPUT control.nacp
                   COMMAND ${NACPTOOL} --create "melonDS switch fast alpha" "Arisotura, RSDuck, Hydr8gon" "0.1" ${CMAKE_BINARY_DIR}/control.nacp)

add_custom_target(melonDS.nro ALL
    ${ELF2NRO} ${CMAKE_BINARY_DIR}/melonDS.elf ${CMAKE_BINARY_DIR}/melonDS.nro --nacp=${CMAKE_BINARY_DIR}/control.nacp
    DEPENDS melonDS.elf control.nacp)
