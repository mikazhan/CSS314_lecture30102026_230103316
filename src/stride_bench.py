import time
import csv
import statistics
import numpy as np

data = np.ones(16777216, dtype=np.float64)
strides = [1, 2, 4, 8, 16, 32, 64, 128]

def read_data(data, stride):
    total = 0.0

    for i in range(0, len(data), stride):
        total += data[i]

    return total

results = []

for stride in strides:
    times = []

    for run in range(10):
        start = time.perf_counter_ns()
        total = read_data(data, stride)
        end = time.perf_counter_ns()

        if run >= 5:
            times.append(end - start)

    median_time = statistics.median(times)
    elements = len(range(0, len(data), stride))
    latency = median_time / elements

    results.append([stride, median_time, latency])

    print(stride, median_time, latency)

with open("results/stride_results.tsv", "w", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["stride", "median_time_ns", "latency_ns_per_element"])
    writer.writerows(results)