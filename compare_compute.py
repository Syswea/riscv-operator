import numpy as np

Q = np.random.rand(8, 4).astype(np.float32)
K = np.random.rand(8, 4).astype(np.float32)
V = np.random.rand(8, 4).astype(np.float32)
O = np.zeros((8, 4), dtype=np.float32)

S = (Q @ K.T) / np.sqrt(4.0)
# 分行作softmax
S_exp = np.exp(S - np.max(S, axis=1, keepdims=True))
S_softmax = S_exp / np.sum(S_exp, axis=1, keepdims=True)
O = S_softmax @ V

print("Q:")
print(Q)
print("K:")
print(K)
print("V:")
print(V)
print("O:")
print(O)