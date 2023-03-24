import numpy as np
import time
from ml_utils.gemm import multiply_naive, multiply_strassen_recursive

target = 1024
a = np.random.randint(1, 11, size=(target, target))
b = np.random.randint(1, 11, size=(target, target))

# start = time.time()
# c = np.dot(a, b)
# print(f"Time for np.dot == {time.time() - start}")

start = time.time()
c = multiply_naive(a, b)
print(f"Time for multiply_naive == {time.time() - start}")

start = time.time()
c = multiply_strassen_recursive(a, b)
print(f"Time for multiply_strassen_recursive == {time.time() - start}")
