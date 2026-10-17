cmake_minimum_required(VERSION 3.22)
project(cuda_wavefront_pathtracer LANGUAGES CXX CUDA)

# ---- options ----
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)   # Hopper (H100 / H800); e.g. -DCMAKE_CUDA_ARCHITECTURES=80 / 86 / 89
endif()
set(CWP_MINB 12 CACHE STRING "__launch_bounds__(128, MINB) of the traversal kernels for pt (0 = no register cap)")
option(CWP_VARIANTS "Also build the register-cap variants of the ablation study (pt_nocap, pt_mb16)" OFF)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

find_package(CUDAToolkit REQUIRED)

# ---- modules ----
# core: math, RNG, sampling (header-only, host + device)
add_library(cwp_core INTERFACE)
target_include_directories(cwp_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/src ${CUDAToolkit_INCLUDE_DIRS})

# scene: mesh, camera, lighting, procedural scenes
add_library(cwp_scene STATIC src/scene/camera.cpp src/scene/synthetic.cpp src/scene/environment.cpp src/io/hdr_image.cpp)
target_link_libraries(cwp_scene PUBLIC cwp_core)

# io: mesh loaders, camera/lighting files, image writers
add_library(cwp_io STATIC src/io/mesh_loader.cpp src/io/json.cpp src/io/scene_files.cpp src/io/image.cpp)
target_link_libraries(cwp_io PUBLIC cwp_scene)

# accel: CPU BVH builder and GPU layouts (device traversal lives in accel/traverse.cuh)
add_library(cwp_accel STATIC src/accel/bvh_builder.cpp src/accel/bvh_layouts.cpp src/accel/tri_pack.cpp)
target_link_libraries(cwp_accel PUBLIC cwp_scene)

# render: wavefront kernels + host driver; one library per register cap
set(CWP_CUDA_FLAGS --use_fast_math -lineinfo --ptxas-options=-v)
function(cwp_add_render name minb)
  add_library(${name} STATIC src/render/wavefront.cu)
  target_link_libraries(${name} PUBLIC cwp_core CUDA::cudart_static)
  target_compile_options(${name} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${CWP_CUDA_FLAGS}>)
  if(minb GREATER 0)
    target_compile_definitions(${name} PRIVATE MINB=${minb})
  endif()
endfunction()

# app: batch renderer CLI
function(cwp_add_pt name minb)
  cwp_add_render(${name}_render ${minb})
  add_executable(${name} src/app/pt.cpp)
  target_link_libraries(${name} PRIVATE ${name}_render cwp_accel cwp_io cwp_scene)
endfunction()

cwp_add_pt(pt ${CWP_MINB})
if(CWP_VARIANTS)
  cwp_add_pt(pt_nocap 0)
  cwp_add_pt(pt_mb16 16)
endif()

# tests: CPU checks of the device BSDF code (chi-square sampling tests, white furnace)
option(CWP_TESTS "Build the CPU BSDF tests (cwp_bsdf_tests)" ON)
if(CWP_TESTS)
  enable_testing()
  add_executable(cwp_bsdf_tests tests/bsdf_tests.cpp)
  target_link_libraries(cwp_bsdf_tests PRIVATE cwp_core)
  add_test(NAME bsdf COMMAND cwp_bsdf_tests)
endif()
