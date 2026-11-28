# The HIP backend: MOSS-Image's kernels for AMD GPUs through ROCm, and the same sources through HIP's NVIDIA headers for
# checking on a CUDA device. Its own project; the engine's build does not include it. Configure once per target (README.md):
#   -DHIP_TARGET="gfx1100;gfx1103;gfx1200;gfx1201"  one or more AMD architectures in one binary:
#                                           RDNA3/RDNA4 use the rocWMMA / native WMMA leaves;
#   -DHIP_TARGET="gfx1103;gfx1035"          a separate fat binary may add gfx1035 (RDNA2, no matrix cores and no bf16
#                                           instructions), whose V_DOT4_I32_I8 / V_DOT2_F32_F16 dot-product leaves need
#                                           -mno-wavefrontsize64; do not combine gfx103x with gfx12 in one build
#   -DHIP_TARGET=nvidia                     scalar leaf, nvcc with HIP's NVIDIA headers
# On Linux the ROCm SDK comes from $ROCM_PATH (-DROCM_PATH=...); on Windows a TheRock SDK, e.g.
#   -DROCM_PATH=<SDK devel root>   (its lib/llvm/bin/clang-cl builds host and device code, its
#   lib/amdhip64.lib links, and its bin/amdhip64*.dll files are copied next to moss-infer-hip).
cmake_minimum_required(VERSION 3.28)
include(${CMAKE_CURRENT_LIST_DIR}/../cmake/version.cmake)  # MOSSINFER_VERSION, from the C header

set(HIP_TARGET "" CACHE STRING "Device targets: a list of gfx1100, gfx1103, gfx1200, gfx1201, gfx1035, or nvidia")
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm root, the directory holding include/hip and lib/llvm")
option(MI_HIP_WERROR "Treat warnings as errors" ON)

# MI_HIP_AMD: the build is for AMD architectures (HIP_TARGET, a list of them), else for an NVIDIA device.
set(MI_HIP_AMD OFF)
if(HIP_TARGET STREQUAL "nvidia")
elseif(HIP_TARGET)
  set(MI_HIP_AMD ON)
  list(REMOVE_DUPLICATES HIP_TARGET)
  foreach(_arch IN LISTS HIP_TARGET)
    if(NOT _arch MATCHES "^(gfx1100|gfx1103|gfx1200|gfx1201|gfx1035)$")
      set(MI_HIP_AMD OFF)
    endif()
  endforeach()
endif()
if(NOT MI_HIP_AMD AND NOT HIP_TARGET STREQUAL "nvidia")
  message(FATAL_ERROR "Set HIP_TARGET to a list of gfx1100, gfx1103, gfx1200, gfx1201, gfx1035, or to nvidia (got '${HIP_TARGET}')")
endif()
if(NOT EXISTS "${ROCM_PATH}/include/hip/hip_runtime.h")
  message(FATAL_ERROR "ROCM_PATH (cache variable or environment) must hold include/hip/hip_runtime.h (got '${ROCM_PATH}')")
endif()
if(MI_HIP_AMD)
  # Windows: clang-cl (MSVC mode, the VS toolchain's C runtime); Linux: ROCm's clang in GNU mode.
  if(WIN32)
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang-cl.exe")
  else()
    set(CMAKE_CXX_COMPILER "${ROCM_PATH}/lib/llvm/bin/clang++")
  endif()
endif()

project(mossinfer_hip VERSION ${MOSSINFER_VERSION} LANGUAGES CXX)

if(MI_HIP_AMD AND WIN32)
  # clang-cl's /Fo object-output flag with several --offload-arch flags makes the ROCm driver emit ONE architecture's
  # code object under every target name in the fat binary: the runtime then tries to load a gfx1103 object on gfx1035
  # and fails. With -o (the GNU-style output flag clang-cl also accepts) each offload target gets its own object.
  # CMake picks /Fo for MSVC-mode compilers, so the compile rule has to be replaced here.
  set(CMAKE_CXX_COMPILE_OBJECT "<CMAKE_CXX_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -o <OBJECT> -c <SOURCE>")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

set(MI_HIP_WARN -Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wno-sign-conversion)
if(MI_HIP_WERROR)
  list(APPEND MI_HIP_WARN -Werror)
endif()
# Host code that replays or extends the kernels' rounding (the CFG and Euler step, the time embedding) rounds every fp32
# operation, as the kernels do.
list(APPEND MI_HIP_WARN -ffp-contract=off)

# Kernel sources. block_kernels.hip is the DiT block (tools/hip_dit_kernel_check checks it against the engine); dit_kernels.hip the
# DiT's other ops; te_kernels.hip the text encoder's; vae_kernels.hip the VAE decoder's.
set(MI_HIP_KERNELS block_kernels.hip dit_kernels.hip te_kernels.hip te_dequant.hip vae_kernels.hip int8_gemm.hip rotate_quantize.hip
                    int8_block.hip int8_attn.hip bf16_gemm.hip lora.hip sage_attn_gfx103x.hip)
set(MI_HIP_KERNEL_HEADERS device_math.h dit_kernels.h gemm_tile.h hip_block.h mma_leaf.h bf16_leaf.h te_kernels.h vae_kernels.h
                           int8_leaf.h int8_block.h int8_gemm.h int8_attn.h bf16_gemm.h lora.h sage_attn_gfx103x.h)
set(MI_HIP_HOST runtime.cpp safetensors.cpp dit.cpp text_encoder.cpp vae.cpp generate.cpp lora.cpp
                ${CMAKE_CURRENT_SOURCE_DIR}/../src/weights/quant/quant_codec.cpp)

if(HIP_TARGET STREQUAL "nvidia")
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures of the NVIDIA build")
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set_source_files_properties(${MI_HIP_KERNELS} PROPERTIES LANGUAGE CUDA)
  list(GET CMAKE_CUDA_ARCHITECTURES 0 _mi_cuda_arch)
  string(REGEX REPLACE "[^0-9].*" "" _mi_cuda_arch "${_mi_cuda_arch}")

  add_library(mossinfer_hip STATIC ${MI_HIP_KERNELS} ${MI_HIP_HOST})
  target_compile_definitions(mossinfer_hip PUBLIC __HIP_PLATFORM_NVIDIA__=1 PRIVATE MI_HIP_TARGET="nvidia"
                                                  MI_HIP_CUDA_ARCH=${_mi_cuda_arch})
  target_include_directories(mossinfer_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
  target_include_directories(mossinfer_hip SYSTEM PUBLIC ${ROCM_PATH}/include ${CUDAToolkit_INCLUDE_DIRS})
  target_compile_options(mossinfer_hip PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
                                               $<$<COMPILE_LANGUAGE:CXX>:${MI_HIP_WARN}>)
  target_link_libraries(mossinfer_hip PUBLIC CUDA::cudart)
else()
  # One fat binary: a code object per architecture. MI_HIP_TARGET names them, comma-separated (runtime.cpp).
  set(MI_HIP_PATHS --rocm-path=${ROCM_PATH} --hip-path=${ROCM_PATH})
  set(MI_HIP_FLAGS ${MI_HIP_PATHS})
  # TheRock places the ocml device library (bitcode) outside the default search path:
  # <rocm>/lib/llvm/amdgcn/bitcode. A full ROCm SDK (Linux) keeps the default /opt/rocm/lib/amdgcn/bitcode.
  if(WIN32)
    list(APPEND MI_HIP_FLAGS --rocm-device-lib-path=${ROCM_PATH}/lib/llvm/amdgcn/bitcode)
  endif()
  foreach(_arch IN LISTS HIP_TARGET)
    list(APPEND MI_HIP_FLAGS --offload-arch=${_arch})
    # RDNA2's HIP default is a 64-lane wave; every kernel here assumes one 32-lane wave per workgroup fragment. The
    # gfx11 targets are wave32 already, so the flag only changes the gfx103x code objects. It is a global driver flag:
    # do not combine gfx103x with gfx12 in one build, whose 256-byte WMMA needs wavefrontsize32.
    if(_arch MATCHES "^gfx103")
      list(APPEND MI_HIP_FLAGS -mno-wavefrontsize64)
    endif()
  endforeach()
  list(JOIN HIP_TARGET "," MI_HIP_TARGET_NAMES)
  set_source_files_properties(${MI_HIP_KERNELS} PROPERTIES LANGUAGE CXX)
  foreach(_src IN LISTS MI_HIP_KERNELS)
    # compat/cmath shadows <cmath> for these HIP sources only, ahead of MSVC's own include directory.
    set_property(SOURCE ${_src} PROPERTY COMPILE_OPTIONS
      -I${CMAKE_CURRENT_SOURCE_DIR}/compat -x hip -std=c++17 ${MI_HIP_FLAGS} ${MI_HIP_WARN})
  endforeach()
  # The RDNA2 attention is tuned with comfy-kitchen's exact flag set: no -ffp-contract=off and a few LLVM knobs.
  # Per-source options replace the warnings list, so keep everything else from MI_HIP_FLAGS.
  set_property(SOURCE sage_attn_gfx103x.hip PROPERTY COMPILE_OPTIONS
    -I${CMAKE_CURRENT_SOURCE_DIR}/compat -x hip -std=c++17 ${MI_HIP_FLAGS}
    -ffast-math -fgpu-flush-denormals-to-zero -fno-offload-uniform-block
    -mllvm=--lsr-drop-solution=1 -mllvm=-enable-post-misched=1
    -mllvm=-amdgpu-early-inline-all=true -mllvm=-amdgpu-function-calls=false
    -mllvm=-amdgpu-max-memory-clause=32 -mllvm=-amdgpu-vgpr-index-mode=1
    -Wall -Wextra -Wno-sign-conversion)
  set_property(SOURCE ${MI_HIP_HOST} PROPERTY COMPILE_OPTIONS ${MI_HIP_WARN})

  add_library(mossinfer_hip STATIC ${MI_HIP_KERNELS} ${MI_HIP_HOST})
  target_compile_definitions(mossinfer_hip PUBLIC __HIP_PLATFORM_AMD__=1 PRIVATE HIP_BLOCK_LEAF_ROCWMMA ROCWMMA_I8_LEAF
                                                  MI_HIP_TARGET="${MI_HIP_TARGET_NAMES}")
  target_include_directories(mossinfer_hip PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/../src)
  target_include_directories(mossinfer_hip SYSTEM PUBLIC ${ROCM_PATH}/include)
  # libamdhip64 from ROCM_PATH; the executables find it there in the build tree (BUILD_RPATH), and the installed
  # moss-infer-hip in /opt/rocm/lib (INSTALL_RPATH, below).
  target_link_directories(mossinfer_hip PUBLIC ${ROCM_PATH}/lib)
  target_link_libraries(mossinfer_hip PUBLIC amdhip64)
  set(CMAKE_BUILD_RPATH ${ROCM_PATH}/lib)

  # Each kernel source's device build on its own, per architecture: the assembly and linked code object
  # check_device.sh reads (<source>.<arch>.s, .co). Same flags as the library's compile. check_device.sh is a
  # bash script, so the check runs on Unix hosts only; the Windows build compiles the same device code through
  # the library's own compile.
  if(NOT WIN32)
  set(MI_HIP_DEVICE_ARGS -x hip -std=c++17 -O3 ${MI_HIP_PATHS} --cuda-device-only -D__HIP_PLATFORM_AMD__=1
      -DHIP_BLOCK_LEAF_ROCWMMA -I${CMAKE_CURRENT_SOURCE_DIR} -isystem ${ROCM_PATH}/include ${MI_HIP_WARN})
  set(_deps "")
  foreach(_h IN LISTS MI_HIP_KERNEL_HEADERS)
    list(APPEND _deps ${CMAKE_CURRENT_SOURCE_DIR}/${_h})
  endforeach()
  set(_checks "")
  # check_device.sh checks WMMA instruction counts, which do not apply to gfx103x's dot-product leaves. Check the
  # matrix-core architectures only; the same sources were still compiled for every HIP_TARGET in the library above.
  set(_check_archs ${HIP_TARGET})
  list(FILTER _check_archs EXCLUDE REGEX "^gfx103")
  # check_device.sh knows the WMMA counts for these four sources. The dot-product and quantizer kernels have no WMMA
  # to count and are verified by their dedicated checks instead.
  set(_check_sources ${MI_HIP_KERNELS})
  list(FILTER _check_sources INCLUDE REGEX "^(block_kernels|dit_kernels|te_kernels|vae_kernels)\\.hip$")
  foreach(_arch IN LISTS _check_archs)
    foreach(_src IN LISTS _check_sources)
      get_filename_component(_name ${_src} NAME_WE)
      set(_out ${_name}.${_arch})
      add_custom_command(OUTPUT ${_out}.s DEPENDS ${_src} ${_deps}
        COMMAND ${CMAKE_CXX_COMPILER} ${MI_HIP_DEVICE_ARGS} --offload-arch=${_arch} -S -o ${_out}.s
                ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
        COMMENT "Device assembly of ${_src} for ${_arch}")
      add_custom_command(OUTPUT ${_out}.co DEPENDS ${_src} ${_deps}
        COMMAND ${CMAKE_CXX_COMPILER} ${MI_HIP_DEVICE_ARGS} --offload-arch=${_arch} --no-gpu-bundle-output -o ${_out}.co
                ${CMAKE_CURRENT_SOURCE_DIR}/${_src}
        COMMENT "Device code object of ${_src} for ${_arch}")
      add_custom_command(OUTPUT ${_out}.check DEPENDS ${_out}.s ${_out}.co ${CMAKE_CURRENT_SOURCE_DIR}/check_device.sh
        COMMAND ${CMAKE_COMMAND} -E env --unset=HIP_BLOCK_WMMA_COUNTS sh ${CMAKE_CURRENT_SOURCE_DIR}/check_device.sh
                ${ROCM_PATH}/lib/llvm/bin ${_arch} ${_out}.s ${_out}.co
        COMMAND ${CMAKE_COMMAND} -E touch ${_out}.check
        COMMENT "check_device.sh ${_name} ${_arch}" VERBATIM)
      list(APPEND _checks ${_out}.check)
    endforeach()
  endforeach()
  add_custom_target(mossinfer_hip_device_check ALL DEPENDS ${_checks})
  endif()
endif()

# moss-infer-hip, the standalone program: the backend plus the engine's host sources it shares with moss-infer (the
# tokenizer, the seed's initial noise, the schedule, the PNG writer), compiled here so that nothing needs CUDA. They are
# the engine's sources, built without this project's warnings; the torch-compatible noise is bit-exact only without
# contraction or fast-math, which clang would otherwise apply.
set(_engine_src ${CMAKE_CURRENT_SOURCE_DIR}/../src)
add_library(mossinfer_hip_host STATIC
  ${_engine_src}/tokenizer/qwen_bpe.cpp ${_engine_src}/tokenizer/unicode.cpp ${_engine_src}/tokenizer/prompt_tokens.cpp
  ${_engine_src}/pipeline/samplers/initial_noise.cpp ${_engine_src}/pipeline/rng_torch_compat.cpp
  ${_engine_src}/pipeline/samplers/schedules.cpp ${_engine_src}/pipeline/png_writer.cpp ${_engine_src}/util/json.cpp
  ${_engine_src}/util/regular_file.cpp ${_engine_src}/util/replaceable_file.cpp)
target_include_directories(mossinfer_hip_host PUBLIC ${_engine_src} ${CMAKE_CURRENT_SOURCE_DIR}/../include)
target_compile_options(mossinfer_hip_host PRIVATE -ffp-contract=off -fno-fast-math)
add_executable(moss-infer-hip tools/moss_infer_hip.cpp)
target_compile_options(moss-infer-hip PRIVATE ${MI_HIP_WARN})
target_link_libraries(moss-infer-hip PRIVATE mossinfer_hip mossinfer_hip_host)

# Package (AMD builds only; the NVIDIA build is for checking): the program and the license files, Linux x86-64. No HIP
# runtime ships: ROCm 7 provides libamdhip64.so.7, found in /opt/rocm/lib (the installed program's RUNPATH) or on the
# loader path.
if(MI_HIP_AMD)
  if(NOT WIN32)
    set_target_properties(moss-infer-hip PROPERTIES INSTALL_RPATH /opt/rocm/lib)
  endif()
  # The program itself then needs no libstdc++ of the build's age; ROCm's libamdhip64 still loads the system's.
  option(MOSSINFER_STATIC_LIBSTDCXX "Link libstdc++ and libgcc statically into moss-infer-hip (release packages)" OFF)
  if(MOSSINFER_STATIC_LIBSTDCXX AND NOT WIN32)
    target_link_options(moss-infer-hip PRIVATE -static-libstdc++ -static-libgcc)
  endif()
  # Windows: the HIP runtime DLLs from the SDK's bin next to the program. The versioned
  # amdhip64*.dll imports rocm_kpack.dll and amd_comgr.dll, so it alone is not enough to load.
  if(WIN32)
    file(GLOB _mi_hip_dlls "${ROCM_PATH}/bin/amdhip64*.dll")
    list(APPEND _mi_hip_dlls "${ROCM_PATH}/bin/rocm_kpack.dll" "${ROCM_PATH}/bin/amd_comgr.dll")
    add_custom_command(TARGET moss-infer-hip POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different ${_mi_hip_dlls} $<TARGET_FILE_DIR:moss-infer-hip>)
  endif()
  install(TARGETS moss-infer-hip RUNTIME DESTINATION bin)
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/../NOTICE
                ${CMAKE_CURRENT_SOURCE_DIR}/../THIRD_PARTY_NOTICES.md ${CMAKE_CURRENT_SOURCE_DIR}/../docs/hip.md
          DESTINATION share/doc/mossinfer)
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _mi_arch)
  if(_mi_arch STREQUAL "amd64")
    set(_mi_arch x86_64)
  endif()
  set(CPACK_PACKAGE_NAME mossinfer-hip)
  set(CPACK_PACKAGE_VERSION ${MOSSINFER_VERSION})
  # The release name for the release's three architectures; a build of fewer names them, so it cannot pass for it.
  set(_mi_hip_archs ${HIP_TARGET})
  list(SORT _mi_hip_archs)
  if(WIN32)
    set(_mi_os windows)
  else()
    set(_mi_os linux)
  endif()
  set(CPACK_PACKAGE_FILE_NAME "mossinfer-${MOSSINFER_VERSION}-${_mi_os}-hip-${_mi_arch}")
  if(NOT _mi_hip_archs STREQUAL "gfx1100;gfx1200;gfx1201")
    list(JOIN _mi_hip_archs "-" _mi_hip_suffix)
    string(APPEND CPACK_PACKAGE_FILE_NAME "-${_mi_hip_suffix}")
  endif()
  set(CPACK_GENERATOR TGZ)
  include(CPack)
endif()

# The checks and tools; the release build compiles them too, and installs only moss-infer-hip.
option(MOSSINFER_HIP_TOOLS "Build the HIP backend's checks and benchmarks" ON)
if(NOT MOSSINFER_HIP_TOOLS)
  return()
endif()

add_executable(hip_attn_probe tools/hip_attn_probe.cpp)
target_compile_options(hip_attn_probe PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_attn_probe PRIVATE mossinfer_hip)

# The int8 attention path against a CPU fp32 reference.
add_executable(hip_int8_attn_probe tools/hip_int8_attn_probe.cpp)
target_compile_options(hip_int8_attn_probe PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_int8_attn_probe PRIVATE mossinfer_hip)

add_executable(hip_check tools/hip_check.cpp)
target_compile_options(hip_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_check PRIVATE mossinfer_hip)

# The INT8 GEMM against a host replay, including partial M/N tiles and a fused LoRA delta.
add_executable(hip_int8_gemm_check tools/hip_int8_gemm_check.cpp)
target_compile_options(hip_int8_gemm_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_int8_gemm_check PRIVATE mossinfer_hip)

# Throughput of the four real GEMM shapes for each tile variant.
add_executable(hip_int8_gemm_bench tools/hip_int8_gemm_bench.cpp)
target_compile_options(hip_int8_gemm_bench PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_int8_gemm_bench PRIVATE mossinfer_hip)

# Fast ConvRot dequantization correctness and speed.
add_executable(hip_dequant_fast_check tools/hip_dequant_fast_check.cpp)
target_compile_options(hip_dequant_fast_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_dequant_fast_check PRIVATE mossinfer_hip)

# Text encoder construction / encode timing breakdown.
add_executable(hip_te_profile tools/hip_te_profile.cpp)
target_compile_options(hip_te_profile PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_te_profile PRIVATE mossinfer_hip)

# LoRA side-branch shapes.
add_executable(hip_lora_info tools/hip_lora_info.cpp)
target_compile_options(hip_lora_info PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_lora_info PRIVATE mossinfer_hip)

# Activation-quantization throughput.
add_executable(hip_rotate_quantize_bench tools/hip_rotate_quantize_bench.cpp)
target_compile_options(hip_rotate_quantize_bench PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_rotate_quantize_bench PRIVATE mossinfer_hip)

# Fused SiTU-GLU + ConvRot quantizer: byte-identical to the two separate ops.
add_executable(hip_fused_quant_check tools/hip_fused_quant_check.cpp)
target_compile_options(hip_fused_quant_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_fused_quant_check PRIVATE mossinfer_hip)

# LoRA GEMM shape throughput.
add_executable(hip_bf16_gemm_bench tools/hip_bf16_gemm_bench.cpp)
target_compile_options(hip_bf16_gemm_bench PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_bf16_gemm_bench PRIVATE mossinfer_hip)

# The tiled bf16 GEMM used by the LoRA side branch and the text encoder's projections.
add_executable(hip_bf16_gemm_check tools/hip_bf16_gemm_check.cpp)
target_compile_options(hip_bf16_gemm_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_bf16_gemm_check PRIVATE mossinfer_hip)

# dit_kernels.h's ops against host replays; the replays round every fp32 operation, as the kernels do.
add_executable(hip_dit_kernel_check tools/hip_dit_kernel_check.cpp)
target_compile_options(hip_dit_kernel_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_dit_kernel_check PRIVATE mossinfer_hip)

# The engine's JSON reader, for hip_dit_run's case files: built as the engine builds it, without this project's warnings.
add_library(mi_hip_engine_json OBJECT ${CMAKE_CURRENT_SOURCE_DIR}/../src/util/json.cpp)
target_include_directories(mi_hip_engine_json SYSTEM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# The DiT's sampling loop from a golden case; CFG and Euler on the host round like the engine's kernels.
add_executable(hip_dit_run tools/hip_dit_run.cpp)
target_compile_options(hip_dit_run PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_dit_run PRIVATE mossinfer_hip mi_hip_engine_json)

# te_kernels.h's ops against host replays of their op order, and the gated delta rule against an fp64 recurrence.
add_executable(hip_te_kernel_check tools/hip_te_kernel_check.cpp)
target_compile_options(hip_te_kernel_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_te_kernel_check PRIVATE mossinfer_hip)

# The text encoder against the official pipeline's golden hidden states; it reads cases.json and the torch noise floor
# with the engine's JSON reader (mi_hip_engine_json above).
add_executable(hip_te_check tools/hip_te_check.cpp)
target_compile_options(hip_te_check PRIVATE ${MI_HIP_WARN})
target_link_libraries(hip_te_check PRIVATE mossinfer_hip mi_hip_engine_json)

# The VAE decoder's kernels against host replays, the exact band-tiled decode against the full one, and (with zlib)
# golden latents against the reference pipeline's PNGs. TheRock SDKs do not ship zlib on Windows: build the check
# without the PNG part there instead of refusing to configure the tools.
find_package(ZLIB QUIET)
add_executable(hip_vae_check tools/hip_vae_check.cpp)
target_compile_options(hip_vae_check PRIVATE ${MI_HIP_WARN})
if(ZLIB_FOUND)
  target_link_libraries(hip_vae_check PRIVATE mossinfer_hip ZLIB::ZLIB)
else()
  # The Windows ROCm SDK has no zlib: build the check without PNG golden-image support (the kernel checks and the
  # band-tiled full-vs-tiled comparison need no PNG files).
  target_compile_definitions(hip_vae_check PRIVATE MI_HIP_VAE_NO_ZLIB=1)
  target_link_libraries(hip_vae_check PRIVATE mossinfer_hip)
  message(STATUS "zlib not found: hip_vae_check builds without PNG golden-image support")
endif()
