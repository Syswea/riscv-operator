# include "tool.h"
# include <riscv_vector.h>

inline void print_rvv(vfloat32m8_t vec, size_t vl) {
    // 1. 获取当前硬件下 m8 向量组的最大元素个数
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    
    // 2. 申请足够大的空间 (也可以使用 alloca 在栈上分配)
    float *buffer = (float *)malloc(vlmax * sizeof(float));
    if (buffer == NULL) return;

    // 3. 将向量数据存入缓冲区
    // 注意：只存储 vl 个元素
    __riscv_vse32_v_f32m8(buffer, vec, vl);

    printf("RVV Vector (vl=%zu): ", vl);
    for (size_t i = 0; i < vl; ++i) {
        printf("%.6f ", buffer[i]);
    }
    printf("\n");

    free(buffer);
}

inline vfloat32m8_t vexp_f32m8(vfloat32m8_t x, size_t vl) {
    // 1. 定义新的阈值：x < -10.0 时 e^x 极小，直接视为 0
    const float LOWER_BOUND = -10.0f; 

    // 2. 生成掩码：判断哪些元素小于 -10.0
    // 当 x < -10.0 时，mask 对应位为 1
    vbool4_t mask = __riscv_vmflt_vf_f32m8_b4(x, LOWER_BOUND, vl);

    // 3. 泰勒级数计算 (1 + x + x^2/2! + ...)
    // 注意：泰勒级数在 x 靠近 -10 时收敛较慢且精度会下降
    vfloat32m8_t result = __riscv_vfadd_vf_f32m8(x, 1.0f, vl);
    vfloat32m8_t term = x;

    for (int n = 2; n <= 10; ++n) {
        term = __riscv_vfmul_vv_f32m8(term, x, vl);
        term = __riscv_vfmul_vf_f32m8(term, 1.0f / (float)n, vl);
        result = __riscv_vfadd_vv_f32m8(result, term, vl);
    }

    // 4. 修正：使用 vfmerge
    // 当 mask 为 1 (即 x < -10.0) 时，将结果强制设为 0.0f
    // 这里的逻辑是：res = mask ? 0.0f : result
    result = __riscv_vfmerge_vfm_f32m8(result, 0.0f, mask, vl);

    return result;
}

// sexp_f32m8 使用cpu计算exp，供调试对比
inline vfloat32m8_t sexp_f32m8(vfloat32m8_t x, size_t vl) {
        // 1. 获取当前硬件下 m8 向量组的最大元素个数
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    
    // 2. 申请足够大的空间 (也可以使用 alloca 在栈上分配)
    float *buffer = (float *)malloc(vlmax * sizeof(float));
    // 报错
    if (buffer == NULL) return __riscv_vundefined_f32m8();

    // 3. 将向量数据存入缓冲区
    // 注意：只存储 vl 个元素
    __riscv_vse32_v_f32m8(buffer, x, vl);

    // 4. 逐元素计算 exp
    for (size_t i = 0; i < vl; ++i) {
        buffer[i] = expf(buffer[i]);
    }
    // 5. 将结果重新加载到向量寄存器
    vfloat32m8_t result = __riscv_vle32_v_f32m8(buffer, vl);

    free(buffer);
    return result;
}

void load(f32 *ram, f32 *hbm, size_t size) {
    size_t vl;
    vfloat32m8_t x = __riscv_vundefined_f32m8();
    for (size_t offset = 0; offset < size; offset += vl) {
        vl = __riscv_vsetvl_e32m8(size - offset);
        x = __riscv_vle32_v_f32m8(hbm + offset, vl);
        __riscv_vse32_v_f32m8(ram + offset, x, vl);
    }
}

void set_init(f32 *O, f32 *m_old, f32 *l) {
    size_t vl;
    vfloat32m8_t x = __riscv_vundefined_f32m8();
    for (size_t offset = 0; offset < br * d; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br * d - offset);
        x = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        __riscv_vse32_v_f32m8(O + offset, x, vl);
    }
    for (size_t offset = 0; offset < br; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br - offset);
        x = __riscv_vfmv_v_f_f32m8(0.0f, vl);
        __riscv_vse32_v_f32m8(l + offset, x, vl);
    }
    for (size_t offset = 0; offset < br; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br - offset);
        x = __riscv_vfmv_v_f_f32m8(-INFINITY, vl);
        __riscv_vse32_v_f32m8(m_old + offset, x, vl);
    }
}
void compute_qk(f32 *S, f32 *Q, f32 *K) {
    size_t vl;
    vfloat32m2_t vs = __riscv_vundefined_f32m2();
    vfloat32m2_t vk = __riscv_vundefined_f32m2();
    for (size_t i = 0; i < br; i++) {
        for (size_t offset = 0; offset < bc; offset += vl) {
            vl = __riscv_vsetvl_e32m2(bc - offset);
            vs = __riscv_vfmv_v_f_f32m2(0.0f, vl);
            for (size_t j = 0; j < d; j ++ ) {
                vk = __riscv_vlse32_v_f32m2(K + offset * d + j, d * sizeof(f32), vl);
                vs = __riscv_vfmacc_vf_f32m2(vs, Q[i * d + j], vk, vl);
            }
            vs = __riscv_vfdiv_vf_f32m2(vs, sqrt(d), vl);
            __riscv_vse32_v_f32m2(S + i * bc + offset, vs, vl);
        }
    }
}

void update_sml(f32 *S, f32 *m_old, f32 *m_new, f32 *l) {
    size_t vl;
    vfloat32m8_t max_s = __riscv_vundefined_f32m8();
    vfloat32m8_t vs = __riscv_vundefined_f32m8();
    vfloat32m8_t pl = __riscv_vundefined_f32m8();
    vfloat32m8_t old = __riscv_vundefined_f32m8();
    for (size_t offset = 0; offset < br; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br - offset);

        pl = __riscv_vle32_v_f32m8(l + offset, vl);
        max_s = __riscv_vle32_v_f32m8(m_old + offset, vl);
        // printf("pl:\n");
        // print_rvv(pl, vl);
        for (size_t j = 0; j < bc; j ++ ) {
            vs = __riscv_vlse32_v_f32m8(S + offset * bc + j, bc * sizeof(f32), vl);
            // vbool4_t mask = __riscv_vmfgt_vv_f32m8_b4(max_s, vs, vl);
            // max_s = __riscv_vmerge_vvm_f32m8(vs, max_s, mask, vl);
            max_s = __riscv_vfmax_vv_f32m8(max_s, vs, vl);
        }
        // pl * exp(m_old - max_s)
        old = __riscv_vle32_v_f32m8(m_old + offset, vl);
        // printf("old:\n");
        // print_rvv(old, vl);
        old = __riscv_vfsub_vv_f32m8(old, max_s, vl);
        // printf("old:\n");
        // print_rvv(old, vl);

        // exp没有实现
        old = sexp_f32m8(old, vl);
        // printf("old:\n");
        // print_rvv(old, vl);
        pl = __riscv_vfmul_vv_f32m8(pl, old, vl);
        // printf("pl:\n");
        // print_rvv(pl, vl);

        for (size_t j = 0; j < bc; j ++ ) {
            vs = __riscv_vlse32_v_f32m8(S + offset * bc + j, bc * sizeof(f32), vl);
            vs = __riscv_vfsub_vv_f32m8(vs, max_s, vl);

            // exp没有实现
            vs = sexp_f32m8(vs, vl);

            pl = __riscv_vfadd_vv_f32m8(pl, vs, vl);
            __riscv_vsse32_v_f32m8(S + offset * bc + j, bc * sizeof(f32), vs, vl);
        }
        print_rvv(pl, vl);
        __riscv_vse32_v_f32m8(l + offset, pl, vl);
        __riscv_vse32_v_f32m8(m_new + offset, max_s, vl);
    }

}

void compute_pv(f32 *O, f32 *S, f32 *V, f32 *m_old, f32 *m_new) {
    size_t vl;
    vfloat32m2_t vo = __riscv_vundefined_f32m2();
    vfloat32m2_t vv = __riscv_vundefined_f32m2();
    for (size_t i = 0; i < br; i ++ ) {
        f32 scale = exp(m_old[i] - m_new[i]);
        for (size_t offset = 0; offset < d; offset += vl) {
            vl = __riscv_vsetvl_e32m2(d - offset);
            vo = __riscv_vle32_v_f32m2(O + i * d + offset, vl);
            vo = __riscv_vfmul_vf_f32m2(vo, scale, vl);
            for (size_t j = 0; j < bc; j ++ ) {
                vv = __riscv_vle32_v_f32m2(V + j * d + offset, vl);
                f32 s = S[i * bc + j];
                vo = __riscv_vfmacc_vf_f32m2(vo, s, vv, vl);
            }
            __riscv_vse32_v_f32m2(O + i * d + offset, vo, vl);
        }
        m_old[i] = m_new[i];
    }
}

void scale(f32 *O, f32 *l) {
    size_t vl;
    vfloat32m8_t vo = __riscv_vundefined_f32m8();
    for (size_t i = 0; i < br; i ++ ) {
        f32 pl = l[i];
        for (size_t offset = 0; offset < d; offset += vl) {
            vl = __riscv_vsetvl_e32m8(d - offset);
            vo = __riscv_vle32_v_f32m8(O + i * d + offset, vl);
            vo = __riscv_vfdiv_vf_f32m8(vo, pl, vl);
            __riscv_vse32_v_f32m8(O + i * d + offset, vo, vl);
        }
    }
}

void store(f32 *ram, f32 *hbm, size_t size) {
    size_t vl;
    vfloat32m8_t x = __riscv_vundefined_f32m8();
    for (size_t offset = 0; offset < size; offset += vl) {
        vl = __riscv_vsetvl_e32m8(size - offset);
        x = __riscv_vle32_v_f32m8(ram + offset, vl);
        __riscv_vse32_v_f32m8(hbm + offset, x, vl);
    }
}

void Memprint(f32 *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%.6f ", data[i]);
        if ((i + 1) % d == 0) printf("\n");
    }
}