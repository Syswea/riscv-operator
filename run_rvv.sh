#!/bin/bash

export PATH="/opt/riscv/bin:$PATH"

BUILD_DIR="build"
# 必须指向 find 命令确定的 sysroot 根目录
SYSROOT="/opt/riscv/sysroot"

if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
fi
mkdir "$BUILD_DIR" && cd "$BUILD_DIR"

echo "--- 开始配置与编译 ---"
cmake .. -DCMAKE_CXX_COMPILER=clang++
make -j$(nproc)

if [ $? -eq 0 ]; then
    echo "--- QEMU 运行结果 ---"
    # -L $SYSROOT 让 QEMU 在 sysroot/lib 和 sysroot/usr/lib 下寻找动态库
    qemu-riscv64 -cpu rv64,v=true,vlen=128 -L "$SYSROOT" ./rvv_test
else
    echo "编译失败。"
    exit 1
fi

cd ..