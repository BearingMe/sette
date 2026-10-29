# C18 native execution path: one compute dispatch, no framework.
#
#   cmake -S src/native -B build/native -G "Visual Studio 17 2022" -A x64
#   cmake --build build/native --config Release
#
# The shader is compiled by DXC (not FXC): the project targets HLSL and DXC per
# M01, and the probe must be able to grow into the 16-bit paths later.
cmake_minimum_required(VERSION 3.20)
project(m_stage_probe LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(m_stage_probe m_stage_probe.cpp)
target_compile_features(m_stage_probe PRIVATE cxx_std_17)
target_compile_definitions(m_stage_probe PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
target_link_libraries(m_stage_probe PRIVATE d3d12 dxgi)

find_program(DXC_EXECUTABLE
    NAMES dxc
    HINTS "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.0.26100.0/x64")

if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR "dxc.exe not found; set -DDXC_EXECUTABLE=<path>")
endif()

set(SHADER_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/gemm_f16_cs.hlsl")
set(SHADER_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gemm_f16_cs.cso")
add_custom_command(
    OUTPUT "${SHADER_OUTPUT}"
    COMMAND "${DXC_EXECUTABLE}" -T cs_6_2 -E main -O3
            "$<IF:$<CONFIG:Debug>,-Zi,-Qstrip_debug>" "${SHADER_SOURCE}" -Fo "${SHADER_OUTPUT}"
    DEPENDS "${SHADER_SOURCE}"
    COMMENT "dxc gemm_f16_cs.hlsl -> gemm_f16_cs.cso")
add_custom_target(m_stage_shaders ALL DEPENDS "${SHADER_OUTPUT}")
add_dependencies(m_stage_probe m_stage_shaders)
