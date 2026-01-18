#pragma once
#include <stdlib.h>
#include <riscv_vector.h>
#include <math.h>
#include <float.h>

typedef std::float32_t f32;

struct BufferHBM {
    size_t N, d, block_size;
    f32 *Q, *K, *V, *O; // shape (N, d)
    BufferHBM(size_t N, size_t d, size_t block_size) : N(N), d(d), block_size(block_size) {
        Q = (f32 *)malloc(sizeof(f32) * N * d);
        K = (f32 *)malloc(sizeof(f32) * N * d);
        V = (f32 *)malloc(sizeof(f32) * N * d);
        O = (f32 *)malloc(sizeof(f32) * N * d);
    }
    ~BufferHBM() {
        free(Q);
        free(K);
        free(V);
        free(O);
    }
};

struct BufferSram {
    size_t N, d, block_size;
    f32 *Q_vector, *O_vector, *S_vector; // shape (1, d)
    f32 *K_block, *V_block; // shape (block_size, d)
    f32 m_glo, l_glo, m_cur, l_cur;
    BufferSram(size_t N, size_t d, size_t block_size) : N(N), d(d), block_size(block_size) {
        Q_vector = (f32 *)malloc(sizeof(f32) * d);
        O_vector = (f32 *)malloc(sizeof(f32) * d);
        S_vector = (f32 *)malloc(sizeof(f32) * block_size);
        K_block = (f32 *)malloc(sizeof(f32) * block_size * d);
        V_block = (f32 *)malloc(sizeof(f32) * block_size * d);
    }
    ~BufferSram() {
        free(Q_vector);
        free(O_vector);
        free(S_vector);
        free(K_block);
        free(V_block);
    }
};

inline vfloat32m2_t rvv_exp_f32m2(vfloat32m2_t x, size_t vl) {
    // 1. 定义常量 (使用更高精度的多项式系数)
    const float log2e  = 1.4426950408f;
    const float ln2_hi = 0.6931457519f;
    const float ln2_lo = 1.4286068203e-6f;
    
    // 5阶泰勒/Remez系数 (1/n!)
    const float c5 = 0.0083333333f; // 1/120
    const float c4 = 0.0416666667f; // 1/24
    const float c3 = 0.1666666667f; // 1/6
    const float c2 = 0.5000000000f; // 1/2
    const float c1 = 1.0000000000f; // 1
    const float c0 = 1.0000000000f; // 1

    // 2. Range Reduction: x = n*ln2 + r
    vfloat32m2_t z = __riscv_vfmul_vf_f32m2(x, log2e, vl);
    // 使用 VXRM_RDN (向负无穷取整) 或简单的 round
    vint32m2_t n = __riscv_vfcvt_x_f_v_i32m2(z, vl); 
    vfloat32m2_t n_f = __riscv_vfcvt_f_x_v_f32m2(n, vl);

    // r = x - n_f * ln2
    vfloat32m2_t r = __riscv_vfnmsac_vf_f32m2(x, ln2_hi, n_f, vl);
    r = __riscv_vfnmsac_vf_f32m2(r, ln2_lo, n_f, vl);

    // 3. 多项式逼近: poly = c0 + r*(c1 + r*(c2 + r*(c3 + r*(c4 + r*c5))))
    // 使用 Horner 方案减少乘法次数，提高指令并行度
    vfloat32m2_t p = __riscv_vfmv_v_f_f32m2(c5, vl);
    p = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(c4, vl), r, p, vl);
    p = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(c3, vl), r, p, vl);
    p = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(c2, vl), r, p, vl);
    p = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(c1, vl), r, p, vl);
    p = __riscv_vfmacc_vf_f32m2(__riscv_vfmv_v_f_f32m2(c0, vl), r, p, vl);

    // 4. 重构结果: res = p * 2^n
    // vfscale 指令是重中之重
    vfloat32m2_t res = __riscv_vfscale_vv_f32m2(p, n, vl);

    return res;
}

void init(f32 &, f32 &, f32 &, f32 &);

void load_vector(f32 *, f32 *, size_t, size_t);
void load_block(f32 *, f32 *, size_t, size_t, size_t);

void store_vector(f32 *, f32 *, size_t, size_t);

void mul_vf(f32 *, f32, size_t);

void softmax(BufferSram &);