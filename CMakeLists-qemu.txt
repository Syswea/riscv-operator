cmake_minimum_required(VERSION 3.10)

# --- 1. 路径与工具链硬编码 (必须在 project 之前) ---
set(RISCV_TOOLCHAIN_ROOT "/home/syswea/Workspace/tools-for-riscv/toolchain")

# 指定交叉编译器绝对路径
set(CMAKE_C_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-clang")
set(CMAKE_CXX_COMPILER "${RISCV_TOOLCHAIN_ROOT}/bin/riscv64-unknown-linux-gnu-clang++")

# 指定系统根目录 (Sysroot)
set(CMAKE_SYSROOT "${RISCV_TOOLCHAIN_ROOT}/sysroot")

# 告知 CMake 我们在进行 Linux 交叉编译
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# --- 2. 项目声明 ---
project(flashatten LANGUAGES C CXX)

# --- 3. 编译选项配置 ---
# 这里指定 RISC-V 特有的架构参数
# -march=rv64gcv: 启用通用指令集(g)、压缩指令(c)和向量扩展(v)
# -mabi=lp64d: 64位双精度浮点 ABI
add_compile_options(
    -march=rv64gcv 
    -mabi=lp64d 
    -mcmodel=medany
    -O3             # 优化等级
    -Wall           # 开启警告
)

# --- 4. 源文件处理 ---
# 搜索 src_c 目录下的所有 C 和 C++ 文件
file(GLOB_RECURSE SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/src_c/*.c"
    "${CMAKE_CURRENT_SOURCE_DIR}/src_c/*.cpp"
)

# --- 5. 生成目标 ---
add_executable(flashatten ${SOURCES})

# --- 6. 链接配置 ---
# 如果未来有特定的库文件 (.a 或 .so)，在此处添加
# target_link_libraries(flashatten PRIVATE ${PROJECT_SOURCE_DIR}/lib/xxx.a)

# 打印信息用于确认
message(STATUS "Compiler: ${CMAKE_C_COMPILER}")
message(STATUS "Sysroot:  ${CMAKE_SYSROOT}")