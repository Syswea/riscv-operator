#include <iostream>
#include <vector>
#include <omp.h>
#include <cstring>
#include "common.h"

// 模拟 DMA: HBM -> SRAM
void load_tile(float* dst_sram, const float* src_hbm, int r_start, int c_start, int r_len, int c_len, int stride) {
    for (int i = 0; i < r_len; i++) {
        // 模拟连续的 DMA 拷贝
        std::memcpy(dst_sram + i * c_len, src_hbm + (r_start + i) * stride + c_start, c_len * sizeof(float));
    }
}

// 模拟 DMA: SRAM -> HBM
void store_tile(float* dst_hbm, const float* src_sram, int r_start, int c_start, int r_len, int c_len, int stride) {
    for (int i = 0; i < r_len; i++) {
        std::memcpy(dst_hbm + (r_start + i) * stride + c_start, src_sram + i * c_len, c_len * sizeof(float));
    }
}

int main() {
    // 1. 初始化 HBM (模拟申请大内存)
    // 使用 std::vector 模拟 HBM 物理内存
    std::vector<float> Q_hbm(N * D);
    std::vector<float> K_hbm(N * D);
    std::vector<float> V_hbm(N * D);
    std::vector<float> O_hbm(N * D, 0.0f);

    // 随机初始化数据... (省略)
    printf("Initializing HBM Memory (N=%d, D=%d)...\n", N, D);
    printf("RVV Configuration: VLEN=128, FP32\n");

    // 2. OpenMP Row Block 并行
    // 外层循环遍历 Q 的块 (Row Blocks)
    int num_row_blocks = (N + Br - 1) / Br;
    int num_col_blocks = (N + Bc - 1) / Bc;

    #pragma omp parallel
    {
        // [关键面试点]
        // 每个线程代表一个核心，拥有独立的 SRAM 空间 (Thread-Local)
        // 避免多个线程竞争同一个 SRAM buffer
        SramBuffer sram; 
        
        // 临时 buffer 用于存放 S (Score) 和 P (Prob)，大小为 Br * Bc
        std::vector<float> S_local(Br * Bc);
        std::vector<float> P_local(Br * Bc);

        #pragma omp for schedule(dynamic)
        for (int i = 0; i < num_row_blocks; i++) {
            int r_start = i * Br;
            int actual_br = std::min(Br, N - r_start);

            // [Step 1] Load Q tile from HBM -> SRAM
            load_tile(sram.Qi, Q_hbm.data(), r_start, 0, actual_br, D, D);
            
            // 初始化 L 和 M (Flash Attention Stats)
            // M 初始化为负无穷, L 初始化为 0
            for(int row=0; row<actual_br; ++row) {
                sram.Mi[row] = -1e9; // 简单起见
                sram.Li[row] = 0.0f;
                // 清零 Output SRAM buffer
                std::memset(sram.Oi + row * D, 0, D * sizeof(float));
            }

            // 内层循环遍历 K, V 的块
            for (int j = 0; j < num_col_blocks; j++) {
                int c_start = j * Bc;
                int actual_bc = std::min(Bc, N - c_start);

                // [Step 2] Load K, V tile from HBM -> SRAM
                load_tile(sram.Kj, K_hbm.data(), c_start, 0, actual_bc, D, D);
                load_tile(sram.Vj, V_hbm.data(), c_start, 0, actual_bc, D, D);

                // [Step 3] Compute S = Q * K^T (RVV Accelerated)
                // 结果存入 S_local
                rvv_matmul_qk(S_local.data(), sram.Qi, sram.Kj, actual_br, actual_bc, D);

                // [Step 4] Compute Softmax Stats & P (RVV Accelerated)
                // 更新 sram.Mi, sram.Li, 计算 P_local
                rvv_softmax_stats(S_local.data(), sram.Mi, sram.Li, P_local.data(), actual_br, actual_bc);

                // [Step 5] Update Output: O = O + P * V (需要实现类似 matmul 的 RVV kernel)
                // 这里为简化展示省略具体 RVV kernel，逻辑是矩阵乘
                // ... rvv_matmul_pv(sram.Oi, P_local.data(), sram.Vj ...);
            }
            
            // [Step 6] Final Rescaling (O = O / L)
            // 在写回 HBM 之前，需要根据最终的 L 对 O 进行归一化
            for(int r=0; r<actual_br; ++r) {
                for(int d=0; d<D; ++d) {
                    sram.Oi[r*D + d] /= sram.Li[r];
                }
            }

            // [Step 7] Store O tile from SRAM -> HBM
            store_tile(O_hbm.data(), sram.Oi, r_start, 0, actual_br, D, D);
        }
    } // End Parallel

    printf("Flash Attention Computation Finished.\n");
    return 0;
}