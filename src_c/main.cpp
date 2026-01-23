# include "tool.h"
// # include <random>
# include <string.h>

// static double rand01() {
//     static thread_local std::mt19937_64 gen(std::random_device{}());
//     static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
//     return dist(gen);
// }

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
    f32 q_data[8][4] = {
        {0.7747557, 0.7294815, 0.23641561, 0.05261101},
        {0.96995074, 0.2957269, 0.13866027, 0.15621738},
        {0.7932213, 0.17625195, 0.02349104, 0.31167844},
        {0.7331366, 0.7484594, 0.5069931, 0.27268022},
        {0.926057,   0.2678152,  0.26401687, 0.85160416},
        {0.9347315,  0.68305504, 0.6407392,   0.23997736},
        {0.57326823, 0.8711417,  0.5988583,  0.8081295 },
        {0.3401206,  0.37908697, 0.5101189,  0.9834948 }
    };
    f32 k_data[8][4] = {
        {0.27160406, 0.9714346,  0.78330225, 0.39577886},
        {0.63183683, 0.93140256, 0.8302526,  0.532477  },
        {0.81285614, 0.95519966, 0.5585761,  0.2365958 },
        {0.98170257, 0.37797412, 0.23655304, 0.03038631},
        {0.8021462,  0.7552725,  0.66006273, 0.7207554 },
        {0.4331118,  0.65901047, 0.13690922, 0.68104666},
        {0.4341585,  0.63583803, 0.8119385,  0.7139002 },
        {0.5222859,  0.25833213, 0.08193354, 0.79936194}
    };
    f32 v_data[8][4] = {
        {0.22032462, 0.79855514, 0.08979022, 0.6840981 },
        {0.20467058, 0.16968994, 0.81256986, 0.21290916},
        {0.9987318,  0.030989,   0.07216651, 0.7262056 },
        {0.3023485,  0.22414535, 0.91888064, 0.28472966},
        {0.99936646, 0.15778916, 0.55904686, 0.5003307 },
        {0.5227179,  0.70045626, 0.17939466, 0.6800838 },
        {0.26209232, 0.61369437, 0.2093189,  0.49825338},
        {0.06711433, 0.68789047, 0.27192786, 0.7134711 }
    };
    memcpy(hbm.Q, q_data, N * d * sizeof(f32));
    memcpy(hbm.K, k_data, N * d * sizeof(f32));
    memcpy(hbm.V, v_data, N * d * sizeof(f32));
    memset(hbm.O, 0, N * d * sizeof(f32));
    // 计算结果
    flash_atten(hbm);
    // 打印结果
    printf("Q:\n");
    Memprint(hbm.Q, N * d);
    printf("K:\n");
    Memprint(hbm.K, N * d);
    printf("V:\n");
    Memprint(hbm.V, N * d);
    printf("O:\n");
    Memprint(hbm.O, N * d);

    return 0;
}