cmake_minimum_required(VERSION 3.15)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

project(RVV_Project CXX)

set(CMAKE_CXX_STANDARD 17)

# --- 关键修复路径 ---
set(RISCV_TOOLCHAIN "/opt/riscv")
set(RISCV_TARGET "riscv64-unknown-linux-gnu")
# 根据你的 find 结果，这是正确的 sysroot 路径
set(RISCV_SYSROOT "${RISCV_TOOLCHAIN}/sysroot")

set(RVV_FLAGS 
    "--target=${RISCV_TARGET}"
    "--gcc-toolchain=${RISCV_TOOLCHAIN}"
    "--sysroot=${RISCV_SYSROOT}"
    "-march=rv64gcv"
    "-mabi=lp64d"
)

add_compile_options(${RVV_FLAGS} -O3 -Wall)

# 显式指定库搜索路径，确保链接器能看到 /usr/lib
add_link_options(
    ${RVV_FLAGS}
    "-fuse-ld=lld"
    "-L${RISCV_SYSROOT}/usr/lib"
    "-L${RISCV_SYSROOT}/lib"
    "-Wl,-rpath-link,${RISCV_SYSROOT}/usr/lib"
    "-Wl,-rpath-link,${RISCV_SYSROOT}/lib"
)

add_executable(rvv_test main.cpp)
target_link_libraries(rvv_test stdc++ m c)