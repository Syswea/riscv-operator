#!/bin/bash

# --- 路径配置 ---
# 根据你的实际安装路径进行调整
RISCV_PATH="/home/syswea/Workspace/tools-for-riscv/toolchain-spike"
SPIKE="$RISCV_PATH/bin/spike"
PK="$RISCV_PATH/riscv64-unknown-elf/bin/pk64"

# --- 输入检查 ---
if [ -z "$1" ]; then
    echo "使用方法: ./run.sh <你的二进制文件路径>"
    echo "示例: ./run.sh build/rvv_app"
    exit 1
fi

APP_PATH=$1
APP_NAME=$(basename $APP_PATH)
LOG_DIR="analysis_logs_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "正在启动 RISC-V 程序分析..."
echo "目标文件: $APP_PATH"
echo "日志目录: $LOG_DIR"
echo "============================================"

# --- Spike 参数解释 ---
# --isa: 指定架构，包含 v 表示开启向量扩展
# -l: 开启指令执行追踪 (Instruction Trace)，记录每一条指令
# --log-commits: 记录寄存器状态的变化
# --ic: 指令缓存配置 (sets:ways:block_size)
# --dc: 数据缓存配置 (sets:ways:block_size)
# --l2: L2 缓存配置
# -s: (pk 参数) 打印系统调用统计信息

$SPIKE --isa=rv64gcv \
       --ic=64:8:64 \
       --dc=64:8:64 \
       --l2=128:8:64 \
       -l \
       --log-commits \
       $PK -s $APP_PATH \
       2> "$LOG_DIR/${APP_NAME}_trace.log" \
       > "$LOG_DIR/${APP_NAME}_stdout.log"

# --- 分析报告生成 ---
echo "分析完成！"
echo "1. 标准输出已保存至: $LOG_DIR/${APP_NAME}_stdout.log"
echo "2. 指令追踪日志已保存至: $LOG_DIR/${APP_NAME}_trace.log (注意：此文件可能非常大)"
echo "3. 系统调用统计已在 stdout 日志末尾显示"

# 提示用户如何查看向量指令执行情况
echo "提示: 你可以使用 'grep vset $LOG_DIR/${APP_NAME}_trace.log' 来查看向量配置指令。"