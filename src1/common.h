#pragma once
#include <vector>
#include <cmath>
#include <riscv_vector.h>

// 矩阵维度配置
const int N = 1024; // Sequence Length
const int D = 512;  // Head Dimension
const int Br = 32;  // Block size for Q (Rows)
const int Bc = 32;  // Block size for K/V (Cols)

// 模拟 SRAM 的结构体
struct SramBuffer {
    float* Qi; // [Br, D]
    float* Kj; // [Bc, D]
    float* Vj; // [Bc, D]
    float* Oi; // [Br, D]
    
    // Flash Attention 统计量
    float* Li; // [Br]  (Row Sum, for normalization)
    float* Mi; // [Br]  (Row Max, for stability)
    
    // 构造函数：在 SRAM 中分配空间
    SramBuffer() {
        Qi = new float[Br * D];
        Kj = new float[Bc * D];
        Vj = new float[Bc * D];
        Oi = new float[Br * D];
        Li = new float[Br];
        Mi = new float[Br];
    }
    
    ~SramBuffer() {
        delete[] Qi; delete[] Kj; delete[] Vj; delete[] Oi;
        delete[] Li; delete[] Mi;
    }
};

// 核心 Kernel 声明
void rvv_matmul_qk(float* S, const float* Q, const float* K, int br, int bc, int d);
void rvv_update_output(float* O, const float* P, const float* V, int br, int bc, int d);
void rvv_softmax_stats(float* S, float* M, float* L, float* P, int br, int bc);