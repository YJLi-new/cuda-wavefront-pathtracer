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
add_library(cwp_scene STATIC src/scene/camera.cpp src/scene/synthetic.cpp src/scene/environment.cpp src/scene/tangents.cpp src/io/hdr_image.cpp)
target_link_libraries(cwp_scene PUBLIC cwp_core)

# io: mesh loaders, camera/lighting files, image writers
add_library(cwp_io STATIC src/io/mesh_loader.cpp src/io/assimp_loader.cpp src/io/json.cpp src/io/scene_files.cpp src/io/image.cpp)
target_link_libraries(cwp_io PUBLIC cwp_scene)

# optional Assimp model loading (obj/ply/fbx/gltf/glb) + stb_image textures, fetched at configure time and pinned:
#   Assimp v5.4.3 (official release tag, BSD-3-Clause), stb at a fixed commit (stb has no release tags; MIT / public domain).
# -DCWP_WITH_ASSIMP=OFF builds without network access; the built-in OBJ/PLY loaders remain.
option(CWP_WITH_ASSIMP "Model loading through Assimp + textures through stb_image (FetchContent)" ON)
if(CWP_WITH_ASSIMP)
  include(FetchContent)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)   # Assimp's bundled zlib still declares a pre-3.5 minimum
  endif()
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(ASSIMP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(ASSIMP_BUILD_ASSIMP_TOOLS OFF CACHE BOOL "" FORCE)
  set(ASSIMP_INSTALL OFF CACHE BOOL "" FORCE)
  set(ASSIMP_NO_EXPORT ON CACHE BOOL "" FORCE)
  set(ASSIMP_WARNINGS_AS_ERRORS OFF CACHE BOOL "" FORCE)
  set(ASSIMP_BUILD_ZLIB ON CACHE BOOL "" FORCE)
  set(ASSIMP_BUILD_DRACO OFF CACHE BOOL "" FORCE)
  set(ASSIMP_INJECT_DEBUG_POSTFIX OFF CACHE BOOL "" FORCE)
  set(ASSIMP_BUILD_ALL_IMPORTERS_BY_DEFAULT OFF CACHE BOOL "" FORCE)
  foreach(fmt OBJ PLY FBX GLTF)
    set(ASSIMP_BUILD_${fmt}_IMPORTER ON CACHE BOOL "" FORCE)
  endforeach()
  FetchContent_Declare(assimp GIT_REPOSITORY https://github.com/assimp/assimp.git GIT_TAG v5.4.3 GIT_SHALLOW TRUE)
  FetchContent_Declare(stb GIT_REPOSITORY https://github.com/nothings/stb.git GIT_TAG 2c980bb59875b0d32144a71867fbdebb2f77cd20)
  FetchContent_MakeAvailable(assimp stb)
  target_compile_definitions(cwp_io PRIVATE CWP_HAVE_ASSIMP=1)
  target_include_directories(cwp_io PRIVATE ${stb_SOURCE_DIR})
  target_link_libraries(cwp_io PRIVATE assimp)
endif()

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
