#pragma once

struct BufferHBM {
    int N, d, block_size;
    float *Q, *K, *V, *O; // shape (N, d)
    BufferHBM(int N, int d, int block_size) : N(N), d(d), block_size(block_size) {
        Q = (float *)malloc(sizeof(float) * N * d);
        K = (float *)malloc(sizeof(float) * N * d);
        V = (float *)malloc(sizeof(float) * N * d);
        O = (float *)malloc(sizeof(float) * N * d);
    }
    ~BufferHBM() {
        free(Q);
        free(K);
        free(V);
        free(O);
    }
};

struct BufferSram {
    int N, d, block_size;
    float *Q_vector, *O_vector, *S_vector; // shape (1, d)
    float *K_block, *V_block; // shape (block_size, d)
    float m_old, m_cur, l_old, l_cur;
    BufferSram(int N, int d, int block_size) : N(N), d(d), block_size(block_size) {
        Q_vector = (float *)malloc(sizeof(float) * d);
        O_vector = (float *)malloc(sizeof(float) * d);
        S_vector = (float *)malloc(sizeof(float) * block_size);
        K_block = (float *)malloc(sizeof(float) * block_size * d);
        V_block = (float *)malloc(sizeof(float) * block_size * d);
    }
    ~BufferSram() {
        free(Q_vector);
        free(O_vector);
        free(S_vector);
        free(K_block);
        free(V_block);
    }
};

void load_vector(float *vector, float *data, int size, int index);
void load_block(float *block, float *data, int size, int index, int block_size);

void compute_vecMULmat(float *vector, float *mat, float *t_vector, int size, int index, int block_size);
void compute_softmax(float *S_vector, float *V_block, float *O_vector, int size, int index, int block_size, float &m_old, float &m_cur, float &l_old, float &l_cur);
void compute_vecMULnum(float *vector, float num, int size, int index);

void store_vector(float *data, float *vector, int size, int index);
