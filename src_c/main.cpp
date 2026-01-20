# include "tool.h"
# include <random>
# include <cstring>

static double rand01() {
    static thread_local std::mt19937_64 gen(std::random_device{}());
    static thread_local std::uniform_real_distribution<double> dist(0.0, 1.0);
    return dist(gen);
}

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
        {0.9300792,  0.43627077, 0.61905825, 0.8356992 },
        {0.60024995, 0.12966536, 0.47848573, 0.94770974},
        {0.34017876, 0.90870374, 0.3669875,  0.20112255},
        {0.33194366, 0.5477741,  0.36230227, 0.07877652},
        {0.73931205, 0.99265957, 0.18152991, 0.02793127},
        {0.42217535, 0.8159349,  0.08559135, 0.62706417},
        {0.0075772,  0.17620902, 0.14791486, 0.44759968},
        {0.7406363,  0.8002156,  0.25091994, 0.6091251 }
    };
    f32 k_data[8][4] = {
        {0.9806126,  0.58587146, 0.67998344, 0.11478161},
        {0.14628688, 0.7473387,  0.9938798,  0.13798195},
        {0.48833475, 0.32327014, 0.29917678, 0.10607103},
        {0.03827352, 0.35011628, 0.44654468, 0.32090482},
        {0.3943458,  0.3837028,  0.37987527, 0.00668148},
        {0.3237722,  0.6530485,  0.28005236, 0.67288595},
        {0.4933567,  0.6225767,  0.28158945, 0.26694658},
        {0.93575525, 0.7564421,  0.32386488, 0.29407203}
    };
    f32 v_data[8][4] = {
        {0.05457192, 0.90922153, 0.8837715,  0.14311694},
        {0.61410505, 0.24608049, 0.46148843, 0.45733446},
        {0.2991076,  0.83822834, 0.0794244,  0.7226192},
        {0.67714673, 0.87308925, 0.72170174, 0.49430883},
        {0.4702012,  0.06092924, 0.5510921,  0.48618543},
        {0.51159966, 0.8040537,  0.32967156, 0.95115274},
        {0.90434855, 0.21568167, 0.59604025, 0.72447795},
        {0.391623,   0.20013233, 0.863353,   0.78851783}
    };
    memcpy(hbm.Q, q_data, N * d * sizeof(f32));
    memcpy(hbm.K, k_data, N * d * sizeof(f32));
    memcpy(hbm.V, v_data, N * d * sizeof(f32));
    memset(hbm.O, 0, N * d * sizeof(f32));
    // 计算结果
    flash_atten(hbm);
    // 打印结果
    printf("Q:\n");
    Oprint(hbm.Q, N * d);
    printf("K:\n");
    Oprint(hbm.K, N * d);
    printf("V:\n");
    Oprint(hbm.V, N * d);
    printf("O:\n");
    Oprint(hbm.O, N * d);

    return 0;
}