#include "tool.h"
#include <stdio.h>
#include <string.h>


void flash_atten(BufferHBM &hbm) {
    f32 *Q = hbm.Q, *K = hbm.K, *V = hbm.V, *O = hbm.O;
    size_t N = hbm.N, d = hbm.d, block_size = hbm.block_size;

    BufferSram sram(N, d, block_size);

    for (size_t i = 0; i < N; i++) {
        load_vector(sram.Q_vector, Q, d, i);
        memset(sram.O_vector, 0, sizeof(f32) * d);

        sram.m_glo = -INFINITY, sram.l_glo = 0.f;

        for (size_t j = 0; j < N; j += block_size) {
            load_block(sram.K_block, K, d, j, block_size);
            load_block(sram.V_block, V, d, j, block_size);

            sram.m_cur = sram.m_glo, sram.l_cur = 0.f;
            softmax(sram);
        }
        mul_vf(sram.O_vector, 1.f / sram.l_cur, d);
        store_vector(sram.O_vector, O, d, i);
    }
}

int main() {

    size_t N = 1024, d = 512;
    size_t block_size = 64;

    BufferHBM hbm(N, d, block_size);
    flash_atten(hbm);

    for (size_t i = 0; i < N; i++) {
        for (size_t j = 0; j < d; j++) {
            printf("%f ", hbm.O[i * d + j]);
        }
        printf("\n");
    }

    return 0;
}