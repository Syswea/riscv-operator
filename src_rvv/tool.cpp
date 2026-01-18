#include "tool.h"

// vector = data[idx]
void load_vector(f32 *vector, f32 *data, size_t d, size_t idx) {
    size_t vl;
    for (size_t offset = 0; offset < d; offset += vl) {
        vl = __riscv_vsetvl_e32m8(d - offset);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(data + idx * d + offset, vl);
        __riscv_vse32_v_f32m8(vector + offset, v, vl);
    }
}

// block = data[idx:idx+block_size]
void load_block(f32 *block, f32 *data, size_t d, size_t idx, size_t block_size) {
    for (size_t i = 0; i < block_size; i++) {
        load_vector(block + i * d, data, d, idx + i);
    }
}

void mul_vf(f32 *vector, f32 mul, size_t len) {
    size_t vl;
    for (size_t offset = 0; offset < len; offset += vl) {
        vl = __riscv_vsetvl_e32m8(len - offset);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(vector + offset, vl);
        v = __riscv_vfmul_vf_f32m8(v, mul, vl);
        __riscv_vse32_v_f32m8(vector + offset, v, vl);
    }
}

void store_vector(f32 *vector, f32 *data, size_t d, size_t idx) {
    size_t vl;
    for (size_t offset = 0; offset < d; offset += vl) {
        vl = __riscv_vsetvl_e32m8(d - offset);
        vfloat32m8_t v = __riscv_vle32_v_f32m8(vector + offset, vl);
        __riscv_vse32_v_f32m8(data + idx * d + offset, v, vl);
    }
}

void softmax(BufferSram &sram) {
    size_t block_size = sram.block_size;
    size_t d = sram.d;
    f32 *Q = sram.Q_vector;
    f32 *O = sram.O_vector;

    f32 *K = sram.K_block;
    f32 *V = sram.V_block;
    
    f32 *S = sram.S_vector;

    size_t vl;
    for (size_t offset = 0; offset < block_size; offset += vl) {
        vl = __riscv_vsetvl_e32m2(block_size - offset);
        vfloat32m2_t s = __riscv_vfmv_v_f_f32m2(0.0f, vl);
        for (size_t i = 0; i < d; i ++ ) {
            f32 q = Q[i];
            vfloat32m2_t k = __riscv_vlse32_v_f32m2(K + offset * d + i, sizeof(f32) * d, vl);
            s = __riscv_vfmacc_vf_f32m2(s, q, s, vl);
        }
        
    }
}
