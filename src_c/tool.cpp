# include "tool.h"

void load(f32 *dst, f32 *src, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

void set_init(f32 *O, f32 *m, f32 *l) {
    for (size_t i = 0; i < br; i++) {
        for (size_t j = 0; j < d; j++) {
            O[i * d + j] = 0;
        }
        m[i] = -INFINITY;
        l[i] = 0;
    }
}

void compute_qk(f32 *S, f32 *Q, f32 *K) {
    for (size_t i = 0; i < br; i++) {
        for (size_t j = 0; j < bc; j++) {
            f32 s = 0;
            for (size_t k = 0; k < d; k++) {
                s += Q[i * d + k] * K[j * d + k];
            }
            S[i * bc + j] = s / sqrt(d);
        }
    }
}

void update_sml(f32 *S, f32 *m_old, f32 *m_new, f32 *l) {
    for (size_t i = 0; i < br; i++) {
        f32 max_s = -INFINITY;
        for (size_t j = 0; j < bc; j++) {
            max_s = fmax(max_s, S[i * bc + j]);
        }
        m_new[i] = fmax(m_old[i], max_s);
        l[i] *= exp(m_old[i] - m_new[i]);
        for (size_t j = 0; j < bc; j++) {
            S[i * bc + j] = exp(S[i * bc + j] - m_new[i]);
            l[i] += S[i * bc + j];
        }
    }
}

void compute_pv(f32 *O, f32 *S, f32 *V, f32 *m_old, f32 *m_new) {
    for (size_t i = 0; i < br; i++) {
        f32 scale_factor = exp(m_old[i] - m_new[i]);

        for (size_t k = 0; k < d; k++) {
            O[i * d + k] *= scale_factor;

            f32 p_v_sum = 0;
            for (size_t j = 0; j < bc; j++) {
                p_v_sum += S[i * bc + j] * V[j * d + k];
            }
            O[i * d + k] += p_v_sum;
        }
    }
    // 更新 m_old
    for (size_t i = 0; i < br; i++) {
        m_old[i] = m_new[i];
    }
}

void scale(f32 *O, f32 *l) {
    for (size_t i = 0; i < br; i++) {
        for (size_t j = 0; j < d; j++) {
            O[i * d + j] /= l[i];
        }
    }
}

void store(f32 *src, f32 *dst, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

void Oprint(f32 *data, size_t size) {
    for (size_t i = 0; i < size; i++) {
        printf("%.6f ", data[i]);
        if ((i + 1) % d == 0) printf("\n");
    }
}