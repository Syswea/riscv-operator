// gemm.cpp
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

using Matrix = std::vector<std::vector<double>>;

Matrix create_matrix(int rows, int cols, double val = 0.0) {
    return Matrix(rows, std::vector<double>(cols, val));
}

void gemm(double alpha, const Matrix& A, const Matrix& B,
          double beta, Matrix& C) {
    int M = A.size();        // rows of A / C
    int K = A[0].size();     // cols of A == rows of B
    int N = B[0].size();     // cols of B / C

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int k = 0; k < K; ++k) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = alpha * sum + beta * C[i][j];
        }
    }
}

// 用于验证的小型测试
void test_gemm() {
    const int M = 3, N = 4, K = 5;
    auto A = create_matrix(M, K);
    auto B = create_matrix(K, N);
    auto C = create_matrix(M, N, 1.0); // 初始化为 1.0

    // 填充简单值
    for (int i = 0; i < M; ++i)
        for (int k = 0; k < K; ++k)
            A[i][k] = i + k;

    for (int k = 0; k < K; ++k)
        for (int j = 0; j < N; ++j)
            B[k][j] = k - j;

    gemm(1.0, A, B, 2.0, C);

    std::cout << "Result C = A*B + 2*C (initial C=1):\n";
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i][j] << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    test_gemm();
    return 0;
}