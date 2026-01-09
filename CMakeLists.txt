# CMakeLists.txt
cmake_minimum_required(VERSION 3.16)
project(GEMMDemo LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 强制使用 Clang++
find_program(CLANGXX clang++)
if(NOT CLANGXX)
    message(FATAL_ERROR "Clang++ not found! Please install clang.")
endif()
set(CMAKE_CXX_COMPILER ${CLANGXX})

# 可选：启用优化和调试信息
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 添加可执行文件
add_executable(gemm_demo gemm.cpp)

# 可选：添加编译选项（例如开启向量化提示）
target_compile_options(gemm_demo PRIVATE
    $<$<CONFIG:RELEASE>:-O3 -march=native -DNDEBUG>
    $<$<CONFIG:DEBUG>:-O0 -g>
)