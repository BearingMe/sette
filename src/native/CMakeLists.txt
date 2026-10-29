# C18/B4 native execution paths: one compute dispatch each, no framework.
#
#   cmake -S src/native -B build/native -G "Visual Studio 17 2022" -A x64
#   cmake --build build/native --config Release
#
# `m_stage_probe` runs one matrix stage from a `MSTAGE01` case (EXP-010).
# `m_image_stage` runs one image stage from a text fixture (EXP-012): the
# bindings, formats, geometry and dispatch come from a captured launch.
#
# The shaders are compiled by DXC (not FXC): the project targets HLSL and DXC per
# M01, and both the f16 and the f64 paths need SM 6.x code generation.
cmake_minimum_required(VERSION 3.20)
project(m_stage_probe LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

add_executable(m_stage_probe m_stage_probe.cpp)
add_executable(m_image_stage m_image_stage.cpp)
foreach(target m_stage_probe m_image_stage)
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_link_libraries(${target} PRIVATE d3d12 dxgi)
endforeach()

find_program(DXC_EXECUTABLE
    NAMES dxc
    HINTS "$ENV{ProgramFiles\(x86\)}/Windows Kits/10/bin/10.0.26100.0/x64")

if(NOT DXC_EXECUTABLE)
    message(FATAL_ERROR "dxc.exe not found; set -DDXC_EXECUTABLE=<path>")
endif()

# shader source|entry point|output name
set(SHADERS
    "gemm_f16_cs.hlsl|main|gemm_f16_cs.cso"
    "image_copy_cs.hlsl|CopyTex|image_copy_tex.cso"
    "image_copy_cs.hlsl|CopyBuf|image_copy_buf.cso"
    "downsample_cs.hlsl|main|downsample_cs.cso"
    "dec0_cs.hlsl|main|dec0_cs.cso")

set(SHADER_OUTPUTS)
foreach(entry IN LISTS SHADERS)
    string(REPLACE "|" ";" fields "${entry}")
    list(GET fields 0 source)
    list(GET fields 1 entrypoint)
    list(GET fields 2 output)
    set(input "${CMAKE_CURRENT_SOURCE_DIR}/${source}")
    set(artifact "${CMAKE_CURRENT_BINARY_DIR}/${output}")
    add_custom_command(
        OUTPUT "${artifact}"
        COMMAND "${DXC_EXECUTABLE}" -T cs_6_2 -E "${entrypoint}" -O3
                "$<IF:$<CONFIG:Debug>,-Zi,-Qstrip_debug>" "${input}" -Fo "${artifact}"
        DEPENDS "${input}"
        COMMENT "dxc ${source}:${entrypoint} -> ${output}")
    list(APPEND SHADER_OUTPUTS "${artifact}")
endforeach()

add_custom_target(m_stage_shaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(m_stage_probe m_stage_shaders)
add_dependencies(m_image_stage m_stage_shaders)
