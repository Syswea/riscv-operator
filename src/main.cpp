#include "tool.h"
#include <stdio.h>

void flash_atten(BufferHBM &hbm) {
    float *Q = hbm.Q, *K = hbm.K, *V = hbm.V, *O = hbm.O;
    int N = hbm.N, d = hbm.d, block_size = hbm.block_size;

    BufferSram sram(N, d, block_size);

    for (int i = 0; i < N; i++) {
        load_vector(sram.Q_vector, Q, d, i);
        memset(sram.O_vector, 0, sizeof(float) * d);
        init(sram.m_old, sram.m_cur, sram.l_old, sram.l_cur);
        for (int j = 0; j < N; j += block_size) {
            load_block(sram.K_block, K, d, j, block_size);
            load_block(sram.V_block, V, d, j, block_size);

            compute_vecMULmat(sram.Q_vector, sram.K_block, sram.S_vector, d, j, block_size);
            compute_softmax(sram.S_vector, sram.V_block, sram.O_vector, d, j, block_size, sram.m_old, sram.m_cur, sram.l_old, sram.l_cur);
        }
        compute_vecMULnum(sram.O_vector, 1.f / sram.l_cur, d, i);
        store_vector(O, sram.O_vector, d, i);
    }
}

int main() {

    int N = 1024, d = 512;
    int block_size = 64;

    BufferHBM hbm(N, d, block_size);
    flash_atten(hbm);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < d; j++) {
            printf("%f ", hbm.O[i * d + j]);
        }
        printf("\n");
    }

    return 0;
}