cmake_minimum_required(VERSION 3.21)
project(DiffSamplerNative VERSION 1.0.6 LANGUAGES CXX RC)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(NOT WIN32)
    message(FATAL_ERROR "The distributable DiffSampler C++ target is Windows-only (Win32 named-pipe worker and DirectML).")
endif()
if(NOT DEFINED ONNXRUNTIME_ROOT OR NOT EXISTS "${ONNXRUNTIME_ROOT}/include/onnxruntime_cxx_api.h")
    message(FATAL_ERROR "Set ONNXRUNTIME_ROOT to the extracted ONNX Runtime DirectML SDK directory.")
endif()
if(NOT EXISTS "${ONNXRUNTIME_ROOT}/include/dml_provider_factory.h")
    message(FATAL_ERROR "The selected ONNX Runtime package does not contain the DirectML provider header.")
endif()
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")
    message(FATAL_ERROR "Missing model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx. Run build_windows.bat to fetch the licensed model.")
endif()

include(FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(WORLD_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WORLD_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
FetchContent_Declare(world
    URL https://github.com/mmorise/World/archive/refs/tags/v1.0.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(world)
if(NOT TARGET world::core)
    message(FATAL_ERROR "The WORLD dependency did not provide its world::core target.")
endif()

set(MODEL_FILE "${CMAKE_CURRENT_SOURCE_DIR}/model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")
set(NOTICE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/model/NOTICE.txt")
set(MANIFEST_FILE "${CMAKE_CURRENT_SOURCE_DIR}/resources/DiffSampler.manifest")
file(TO_CMAKE_PATH "${MANIFEST_FILE}" MANIFEST_FILE_RC)
configure_file(resources/app.rc.in "${CMAKE_CURRENT_BINARY_DIR}/app_resource.rc" @ONLY)

add_executable(DiffSampler
    cpp/diffsampler.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/app_resource.rc"
)
target_compile_features(DiffSampler PRIVATE cxx_std_17)
target_include_directories(DiffSampler PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/cpp"
    "${ONNXRUNTIME_ROOT}/include"
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/session"
    "${ONNXRUNTIME_ROOT}/include/onnxruntime/core/providers/dml"
)
target_link_directories(DiffSampler PRIVATE "${ONNXRUNTIME_ROOT}/lib")
target_link_libraries(DiffSampler PRIVATE world::core onnxruntime d3d12 dxgi)
target_compile_definitions(DiffSampler PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE _USE_MATH_DEFINES)

if(MSVC)
    target_compile_options(DiffSampler PRIVATE /W4 /permissive- /EHsc /utf-8 /guard:cf)
    # The application manifest is already embedded once by resources/app.rc.in.
    # Disable LINK.exe auto-generation to avoid duplicate RT_MANIFEST resources (CVTRES Cvt1100).
    target_link_options(DiffSampler PRIVATE /DYNAMICBASE /NXCOMPAT /HIGHENTROPYVA /guard:cf /Brepro /MANIFEST:NO)
    set_property(TARGET DiffSampler PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # ONNX Runtime's prebuilt DLLs use the dynamic MSVC runtime. Bundle the
    # matching redistributable DLLs so end users do not need a separate install.
    include(InstallRequiredSystemLibraries)
endif()

set_target_properties(DiffSampler PROPERTIES OUTPUT_NAME "DiffSampler")
set(DIST_DIR "${CMAKE_BINARY_DIR}/dist")
add_custom_command(TARGET DiffSampler POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:DiffSampler>" "${DIST_DIR}/DiffSampler.exe"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${ONNXRUNTIME_ROOT}/bin" "${DIST_DIR}"
    COMMAND ${CMAKE_COMMAND} -E make_directory "${DIST_DIR}/model"
    COMMAND ${CMAKE_COMMAND} -E copy "${MODEL_FILE}" "${DIST_DIR}/model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
    COMMAND ${CMAKE_COMMAND} -E copy "${NOTICE_FILE}" "${DIST_DIR}/model/NOTICE.txt"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/README.md" "${DIST_DIR}/README.txt"
    COMMAND ${CMAKE_COMMAND} -E copy "${world_SOURCE_DIR}/LICENSE.txt" "${DIST_DIR}/WORLD-LICENSE.txt"
    VERBATIM
)
if(MSVC AND CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
    foreach(RUNTIME_DLL IN LISTS CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS)
        add_custom_command(TARGET DiffSampler POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${RUNTIME_DLL}" "${DIST_DIR}"
            VERBATIM
        )
    endforeach()
endif()

message(STATUS "DiffSampler native build")
message(STATUS "  ONNX Runtime: ${ONNXRUNTIME_ROOT}")
message(STATUS "  WORLD source: ${world_SOURCE_DIR} (v1.0.1)")
message(STATUS "  Model: external sidecar file in dist/model")

if(EXISTS "${ONNXRUNTIME_ROOT}/LICENSE")
    add_custom_command(TARGET DiffSampler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/LICENSE" "${DIST_DIR}/ONNXRUNTIME-LICENSE.txt"
        VERBATIM
    )
endif()
if(EXISTS "${ONNXRUNTIME_ROOT}/ThirdPartyNotices.txt")
    add_custom_command(TARGET DiffSampler POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${ONNXRUNTIME_ROOT}/ThirdPartyNotices.txt" "${DIST_DIR}/ONNXRUNTIME-THIRD-PARTY-NOTICES.txt"
        VERBATIM
    )
endif()
