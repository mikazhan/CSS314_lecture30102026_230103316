import time
import json
import statistics
import numpy as np

M = 544

A = np.arange(M * M, dtype=np.float64).reshape(M, M)
B = np.arange(M * M, dtype=np.float64).reshape(M, M)

def naive(A, B):
    C = np.zeros((M, M))

    for i in range(M):
        for j in range(M):
            for k in range(M):
                C[i, j] += A[i, k] * B[k, j]

    return C

def interchanged(A, B):
    C = np.zeros((M, M))

    for i in range(M):
        for k in range(M):
            a = A[i, k]

            for j in range(M):
                C[i, j] += a * B[k, j]

    return C

naive_times = []
interchanged_times = []

for run in range(3):
    start = time.perf_counter()
    C1 = naive(A, B)
    naive_times.append(time.perf_counter() - start)

    start = time.perf_counter()
    C2 = interchanged(A, B)
    interchanged_times.append(time.perf_counter() - start)

naive_median = statistics.median(naive_times)
interchanged_median = statistics.median(interchanged_times)

speedup = naive_median / interchanged_median

results = {
    "M": M,
    "naive_times": naive_times,
    "interchanged_times": interchanged_times,
    "naive_median": naive_median,
    "interchanged_median": interchanged_median,
    "speedup": speedup,
    "equal": bool(np.allclose(C1, C2))
}

with open("results/loop_interchange_practice.json", "w") as f:
    json.dump(results, f, indent=4)

print("Equal:", results["equal"])
print("Naive median:", naive_median)
print("Interchanged median:", interchanged_median)
print("Speedup:", speedup)