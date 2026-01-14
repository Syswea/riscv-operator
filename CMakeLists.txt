cmake_minimum_required(VERSION 3.10)
project(riscv_flash_attn)

set(CMAKE_CXX_STANDARD 17)

# 开启 OpenMP
find_package(OpenMP REQUIRED)

# 编译器标志
# -march=rv64gcv: 启用 64位 RISC-V, Graphics(通用), Compressed, Vector 扩展
# -menable-experimental-extensions: 针对某些旧版 Clang 需要
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=rv64gcv -fopenmp")

include_directories(src)

add_executable(flash_attn src/main.cpp src/kernels.cpp)
target_link_libraries(flash_attn OpenMP::OpenMP_CXX)