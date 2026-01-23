#!/bin/bash

# --- 1. 基础路径配置 ---
RISCV_TOOLCHAIN_ROOT="/home/syswea/Workspace/tools-for-riscv/toolchain"
BINARY="./demo/flashatten"
QEMU_BIN="$RISCV_TOOLCHAIN_ROOT/bin/qemu-riscv64"
SYSROOT="$RISCV_TOOLCHAIN_ROOT/sysroot"

# --- 2. 运行环境参数设置 (关键部分) ---

# [CPU 参数] 
# rv64: 64位架构
# v=true: 开启向量扩展 (RVV)
# vlen=256: 设置硬件向量长度为 256位 (可根据需要改为 128, 512, 1024等)
# vext_spec=v1.0: 确保使用 RVV 1.0 标准 (如果你的代码是旧版，可改为 v0.7.1)
CPU_OPTS="rv64,v=true,vlen=256,vext_spec=v1.0"

# [OpenMP 参数]
# 设置并行线程数
export OMP_NUM_THREADS=4

# --- 3. 检查文件 ---
if [ ! -f "$BINARY" ]; then
    echo "错误: 找不到可执行文件 $BINARY 请先运行 ./build.sh"
    exit 1
fi

# --- 4. 执行运行 ---
echo "======================================="
echo "正在运行: $BINARY"
echo "配置信息: VLEN=$VLEN (via CPU $CPU_OPTS)"
echo "线程设置: OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "======================================="

# 执行命令说明：
# -L: 指定动态链接库搜索路径 (sysroot)
# -cpu: 注入上面定义的硬件特性参数
$QEMU_BIN -L $SYSROOT -cpu "$CPU_OPTS" $BINARY "$@"

# --- 5. 状态检查 ---
RET=$?
if [ $RET -eq 0 ]; then
    echo "---------------------------------------"
    echo "运行成功"
else
    echo "---------------------------------------"
    echo "运行失败，退出码: $RET"
fi