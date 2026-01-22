# pragma once
# include <cmath>
# include <cstdlib>
# include <limits>
# include <cstdio>

# ifndef INFINITY
#  define INFINITY std::numeric_limits<float>::infinity()
# endif

const size_t N = 8, d = 4, br = 4, bc = 4;

typedef float f32;

struct HBM {
    size_t N, d;
    f32 *Q, *K, *V, *O;
    HBM(size_t N, size_t d) : N(N), d(d) {
        Q = (f32 *)malloc(N * d * sizeof(f32));
        K = (f32 *)malloc(N * d * sizeof(f32));
        V = (f32 *)malloc(N * d * sizeof(f32));
        O = (f32 *)malloc(N * d * sizeof(f32));
    }
    ~HBM() {
        free(Q);
        free(K);
        free(V);
        free(O);
    }
};

struct SRAM {
    size_t br, bc, d;
    f32 *Q, *K, *V, *O, *S;
    f32 *m_old, *l, *m_new;
    SRAM(size_t br, size_t bc, size_t d) : br(br), bc(bc), d(d) {
        Q = (f32 *)malloc(br* d * sizeof(f32));
        O = (f32 *)malloc(br* d * sizeof(f32));
        K = (f32 *)malloc(bc * d * sizeof(f32));
        V = (f32 *)malloc(bc * d * sizeof(f32));
        S = (f32 *)malloc(br * bc * sizeof(f32));
        m_old = (f32 *)malloc(br * sizeof(f32));
        m_new = (f32 *)malloc(br * sizeof(f32));
        l = (f32 *)malloc(br * sizeof(f32));
    }
    ~SRAM() {
        free(Q);
        free(K);
        free(V);
        free(S);
        free(O);
        free(m_old);
        free(l);
        free(m_new);
    }
};

void load(f32 *, f32 *, size_t );
void set_init(f32 *, f32 *, f32 *);
void compute_qk(f32 *, f32 *, f32 *);
void update_sml(f32 *, f32 *, f32 *, f32 *);
void compute_pv(f32 *, f32 *, f32 *, f32 *, f32 *);
void scale(f32 *, f32 *);
void store(f32 *, f32 *, size_t );

void Memprint(f32 *, size_t);
