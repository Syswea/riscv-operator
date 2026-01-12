#include <riscv_vector.h>
#include <stdint.h>

// 外部声明 printf，避免包含 <cstdio> 及其背后的 <bits/c++config.h>
extern "C" int printf(const char* format, ...);

// RVV 实现的向量加法：C = A + B
void vec_add_rvv(const float* a, const float* b, float* c, size_t n) {
    for (size_t vl; n > 0; n -= vl, a += vl, b += vl, c += vl) {
        // 设置当前循环的向量长度 (e32: 32位浮点, m1: 1个寄存器组)
        vl = __riscv_vsetvl_e32m1(n);

        // 加载数据
        vfloat32m1_t va = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vb = __riscv_vle32_v_f32m1(b, vl);

        // 向量加法
        vfloat32m1_t vc = __riscv_vfadd_vv_f32m1(va, vb, vl);

        // 存储数据
        __riscv_vse32_v_f32m1(c, vc, vl);
    }
}

int main() {
    const size_t N = 64;
    float a[N], b[N], c[N];

    // 初始化数据
    for (size_t i = 0; i < N; ++i) {
        a[i] = (float)i;
        b[i] = (float)i * 2.0f;
        c[i] = 0.0f;
    }

    printf("Starting RVV Vector Addition...\n");
    vec_add_rvv(a, b, c, N);

    // 验证部分结果
    printf("Result check:\n");
    for (size_t i = 0; i < 5; ++i) {
        printf("  c[%zu] = %.1f (Expected: %.1f)\n", i, c[i], a[i] + b[i]);
    }

    if (c[N-1] == a[N-1] + b[N-1]) {
        printf("Success!\n");
    } else {
        printf("Failed!\n");
    }

    return 0;
}