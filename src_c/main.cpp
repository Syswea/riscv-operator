# include "tool.h"

void flash_atten(HBM &hbm) {
    size_t N = hbm.N, d = hbm.d;
    SRAM sram(br, bc, d);

    for (size_t i = 0; i < N; i += br) {
        load(sram.Q, hbm.Q + i * d, br * d);
        set_init(sram.O, sram.m_old, sram.l);
        for (size_t j = 0; j < N; j += bc) {
            load(sram.K, hbm.K + j * d, bc * d);
            load(sram.V, hbm.V + j * d, bc * d);

            compute_qk(sram.S, sram.Q, sram.K);
            update_sml(sram.S, sram.m_old, sram.m_new, sram.l);
            compute_pv(sram.O, sram.S, sram.V, sram.m_old, sram.m_new);
        }
        scale(sram.O, sram.l);
        store(sram.O, hbm.O + i * d, br * d);
    }
}

int main() {
    HBM hbm(N, d);
    // 写入HBM
    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < d; j++) {
            hbm.Q[i * d + j] = i * d + j;
            hbm.K[i * d + j] = i * d + j;
            hbm.V[i * d + j] = i * d + j;
            hbm.O[i * d + j] = 0;
        }
    }
    // 计算结果
    flash_atten(hbm);
    // 打印结果
    fprint(hbm.O, N * d);

    return 0;
}