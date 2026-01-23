#!/bin/bash

# --- 配置区 ---
BUILD_DIR="demo"

# --- 第一步：清理旧的构建目录 ---
if [ -d "$BUILD_DIR" ]; then
    echo "提示: 检测到已存在的 $BUILD_DIR 文件夹，正在删除并重新创建..."
    rm -rf "$BUILD_DIR"
fi

mkdir "$BUILD_DIR"
cd "$BUILD_DIR"

# --- 第二步：运行 CMake 构建 Makefile ---
# 因为你已经在 CMakeLists.txt 中指定了编译器路径，这里直接运行即可
echo "提示: 正在使用 CMake 生成 Makefile..."
cmake ..

# 检查 CMake 是否成功
if [ $? -ne 0 ]; then
    echo "错误: CMake 配置失败，请检查 CMakeLists.txt 中的路径是否正确。"
    exit 1
fi

# --- 第三步：执行编译 ---
# 使用 -j$(nproc) 利用所有 CPU 核心加速编译
echo "提示: 开始编译项目..."
make -j$(nproc)

# --- 第四步：检查编译结果 ---
if [ $? -eq 0 ]; then
    echo "======================================="
    echo "构建成功！"
    echo "二进制文件位置: $BUILD_DIR/flashatten"
    echo "======================================="
    
    # 可选：查看生成的 ELF 文件属性
    echo "目标架构信息:"
    ls -l flashatten
    file flashatten
else
    echo "======================================="
    echo "错误: 编译过程中出现问题。"
    echo "======================================="
    exit 1
fi