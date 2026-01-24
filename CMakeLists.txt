cmake_minimum_required(VERSION 3.10)

# 1. 定义项目及支持的语言
project(RISCV_RVV_Project C CXX)

# 2. 路径配置 (请确保此路径正确)
set(RISCV_TOOLCHAIN_ROOT "/home/syswea/Workspace/tools-for-riscv/toolchain-spike")
set(RISCV_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")

# 3. 编译器三元组定义
set(TARGET_TRIPLE "riscv64-unknown-linux-gnu")

# 4. 指定编译器二进制路径
set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/clang++")

# 5. 核心编译标志
# -march=rv64gcv: 启用向量扩展 (RVV)
# --sysroot: 必须指向包含标准库头文件和库文件的目录
set(COMMON_FLAGS "--target=${TARGET_TRIPLE} --sysroot=${RISCV_SYSROOT} -march=rv64gcv -mabi=lp64d")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${COMMON_FLAGS} -O3 -Wall")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMMON_FLAGS} -O3 -Wall")

# 6. 链接标志 (解决 __gxx_personality_v0 错误的关键)
# -static: 静态链接，Spike + pk 环境运行的必要条件
# -lstdc++: 手动链接 GNU C++ 标准库
# -lm: 链接数学库
# -lgcc_s -lgcc_eh: 解决异常处理相关的未定义引用 (personality_v0)
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -lstdc++ -lm -lgcc -lgcc_eh")

# 7. 源码管理
# 包含源文件目录的头文件
include_directories(src_rvv)

# 递归寻找 src_rvv 目录下的所有源文件
file(GLOB_RECURSE SOURCES "src_rvv/*.cpp" "src_rvv/*.c")

# 8. 生成可执行文件
if(SOURCES)
    add_executable(rvv_app ${SOURCES})
else()
    message(FATAL_ERROR "No source files found in src_rvv directory!")
endif()

# 调试信息打印
message(STATUS "----------------------------------------------------")
message(STATUS "RISC-V Toolchain Root: ${RISCV_TOOLCHAIN_ROOT}")
message(STATUS "Using Sysroot: ${RISCV_SYSROOT}")
message(STATUS "Target Triple: ${TARGET_TRIPLE}")
message(STATUS "----------------------------------------------------")