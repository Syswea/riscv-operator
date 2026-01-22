# include "tool.h"
# include <riscv_vector.h>



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
    vfloat32m8_t x = __riscv_vxor_vv_f32m8(x, x, 0); // set to 0
    for (size_t offset = 0; offset < br * d; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br * d - offset);
        __riscv_vse32_v_f32m8(O + offset, x, vl);
    }
    for (size_t offset = 0; offset < br; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br - offset);
        __riscv_vse32_v_f32m8(l + offset, x, vl);
    }
    x = __riscv_vfmv_v_f_f32m8(-INFINITY, 0);
    for (size_t offset = 0; offset < br; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br - offset);
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
            vs = __riscv_vxor_vv_f32m2(vs, vs, 0); // set to 0
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
    vbool4_t mask = __riscv_vundefined_b4();
    for (size_t offset = 0; offset < br; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br - offset);

        pl = __riscv_vle32_v_f32m8(l + offset, vl);
        max_s = __riscv_vle32_v_f32m8(m_old + offset, vl);

        for (size_t j = 0; j < bc; j ++ ) {
            vs = __riscv_vlse32_v_f32m8(S + offset * bc + j, bc * sizeof(f32), vl);
            mask = __riscv_vfgt_vv_f32m8_b4(max_s, vs, vl);
            max_s = __riscv_vmerge_vvm_f32m8(vs, max_s, mask, vl);
        }

        for (size_t j = 0; j < bc; j ++ ) {
            vs = __riscv_vlse32_v_f32m8(S + offset * bc + j, bc * sizeof(f32), vl);
            vs = __riscv_vfsub_vv_f32m8(vs, max_s, vl);

            // exp没有实现
            vs = __riscv_vexp_v_f32m8(vs, vl);

            pl = __riscv_vfadd_vv_f32m8(pl, vs, vl);
            __riscv_vse32_v_f32m8(S + offset * bc + j, vs, vl);
        }
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
    vfloat32m8_t vl = __riscv_vundefined_f32m8();
    for (size_t offset = 0; offset < br * d; offset += vl) {
        vl = __riscv_vsetvl_e32m8(br * d - offset);
        vo = __riscv_vle32_v_f32m8(O + offset, vl);
        vo = __riscv_vfdiv_vf_f32m8(vo, l[offset / d], vl);
        __riscv_vse32_v_f32m8(O + offset, vo, vl);
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