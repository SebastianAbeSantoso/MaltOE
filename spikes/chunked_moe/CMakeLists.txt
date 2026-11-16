cmake_minimum_required(VERSION 3.21)
project(molt_spike_chunked_moe CXX C)

# Point this at a llama.cpp checkout at the commit pinned in docs/feasibility-report.md.
set(LLAMA_CPP_DIR "" CACHE PATH "Path to the llama.cpp source tree")
if (NOT LLAMA_CPP_DIR OR NOT EXISTS "${LLAMA_CPP_DIR}/ggml/CMakeLists.txt")
    message(FATAL_ERROR "Set -DLLAMA_CPP_DIR=<path to llama.cpp checkout>")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Turn backends on from the command line, for example -DGGML_CUDA=ON or -DGGML_VULKAN=ON.
set(GGML_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory("${LLAMA_CPP_DIR}/ggml" ggml)

add_executable(chunked_moe chunked_moe.cpp)
target_link_libraries(chunked_moe PRIVATE ggml)

# The MSVC STL needs C++ exception semantics enabled; without this it warns (C4530) when std::vector is used.
if (MSVC)
    target_compile_options(chunked_moe PRIVATE /EHsc)
endif()
