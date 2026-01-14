#include "common.h"
#include <limits>
#include <algorithm>

// 1. 计算 S = Q * K^T 使用 RVV
// Q: [Br, D], K: [Bc, D], S: [Br, Bc]
// 这里的优化点：使用向量累加指令
void rvv_matmul_qk(float* S, const float* Q, const float* K, int br, int bc, int d) {
    size_t vl;
    for (int i = 0; i < br; i++) {
        for (int j = 0; j < bc; j++) {
            float sum = 0.0f;
            const float* q_ptr = Q + i * d;
            const float* k_ptr = K + j * d;
            
            // 使用 m1 寄存器组 (32bit float, VLEN=128 => 4 floats per vector)
            vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
            
            for (int k = 0; k < d; k += vl) {
                vl = __riscv_vsetvl_e32m1(d - k);
                
                vfloat32m1_t v_q = __riscv_vle32_v_f32m1(q_ptr + k, vl);
                vfloat32m1_t v_k = __riscv_vle32_v_f32m1(k_ptr + k, vl);
                
                // Fused Multiply-Add: v_sum += v_q * v_k
                v_sum = __riscv_vfmacc_vv_f32m1(v_sum, v_q, v_k, vl);
            }
            
            // 将向量寄存器中的部分和规约 (Reduction) 为标量
            vfloat32m1_t v_red = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, v_sum, __riscv_vsetvlmax_e32m1());
            sum = __riscv_vfmv_f_s_f32m1_f32(v_red);
            
            S[i * bc + j] = sum;
        }
    }
}

// 2. Online Softmax 核心逻辑的 RVV 辅助
// 更新 Max, 计算 Exp, 更新 Sum
// S: [Br, Bc] 输入分数
// M: [Br] 当前最大值
// L: [Br] 当前分母和
// P: [Br, Bc] 输出概率矩阵
void rvv_softmax_stats(float* S, float* M, float* L, float* P, int br, int bc) {
    // 简化版实现，针对每一行进行处理
    // 注意：在真实的 Flash Attention 中，M 和 L 是动态更新的
    // 这里展示如何用 RVV 快速计算一行内的 Max 和 ExpSum
    
    for (int i = 0; i < br; i++) {
        float* row_s = S + i * bc;
        float* row_p = P + i * bc;
        
        // Step A: Find Local Max in the row block
        float local_max = -std::numeric_limits<float>::infinity();
        size_t vl;
        vfloat32m1_t v_max = __riscv_vfmv_v_f_f32m1(local_max, __riscv_vsetvlmax_e32m1());
        
        for (int j = 0; j < bc; j += vl) {
            vl = __riscv_vsetvl_e32m1(bc - j);
            vfloat32m1_t v_s = __riscv_vle32_v_f32m1(row_s + j, vl);
            v_max = __riscv_vfmax_vv_f32m1(v_max, v_s, vl);
        }
        vfloat32m1_t v_red_max = __riscv_vfredmax_vs_f32m1_f32m1(v_max, v_max, __riscv_vsetvlmax_e32m1());
        local_max = __riscv_vfmv_f_s_f32m1_f32(v_red_max);
        
        // 更新全局 Max (M[i]) 的逻辑需结合历史值，这里为演示简化为 Block 内处理
        // 实际上 Flash Attention 需要比较 M_prev 和 local_max
        float m_prev = M[i];
        float m_new = std::max(m_prev, local_max);
        M[i] = m_new;
        
        // Step B: Calculate Exp and Sum
        float sum_exp = 0.0f;
        vfloat32m1_t v_sum = __riscv_vfmv_v_f_f32m1(0.0f, __riscv_vsetvlmax_e32m1());
        
        for (int j = 0; j < bc; j += vl) {
            vl = __riscv_vsetvl_e32m1(bc - j);
            vfloat32m1_t v_s = __riscv_vle32_v_f32m1(row_s + j, vl);
            
            // P[j] = exp(S[j] - m_new)
            vfloat32m1_t v_diff = __riscv_vfsub_vf_f32m1(v_s, m_new, vl);
            
            // 注意：Clang/LLVM 通常没有直接的 vfexp 指令，需要数学库或泰勒展开近似
            // 这里为了编译通过，使用 std::exp 的标量回退，或者假设有 __riscv_vfexp (某些扩展库)
            // 真正高性能实现这里会用多项式拟合 exp
            // 这里我们用一个简化的标量循环模拟向量化操作的占位
            for(int k=0; k<vl; ++k) {
                row_p[j+k] = std::exp(row_s[j+k] - m_new);
            }
            // 重新加载 P 来做 sum (如果上面是向量化计算的话)
            vfloat32m1_t v_p = __riscv_vle32_v_f32m1(row_p + j, vl);
            v_sum = __riscv_vfadd_vv_f32m1(v_sum, v_p, vl);
        }
        
        vfloat32m1_t v_red_sum = __riscv_vfredusum_vs_f32m1_f32m1(v_sum, v_sum, __riscv_vsetvlmax_e32m1());
        float local_sum = __riscv_vfmv_f_s_f32m1_f32(v_red_sum);
        
        // Update L[i] with rescaling logic (Flash Attention formula)
        // L_new = L_prev * exp(m_prev - m_new) + local_sum
        L[i] = L[i] * std::exp(m_prev - m_new) + local_sum;
    }
}