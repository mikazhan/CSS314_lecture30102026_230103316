import multiprocessing as mp
import numpy as np
import ctypes
import time
import math
import csv
import json
import os

N = 1088
workers = [1, 2, 4, 8]
iterations = 10
seed = 18129


def stencil_worker(source, output, n, start, end, repeats):
    a = np.frombuffer(source, dtype=np.float64).reshape(n, n)
    b = np.frombuffer(output, dtype=np.float64).reshape(n, n)

    for _ in range(repeats):
        b[start:end, 1:-1] = 0.2 * (
            a[start - 1:end - 1, 1:-1]
            + a[start + 1:end + 1, 1:-1]
            + a[start:end, :-2]
            + a[start:end, 2:]
            + a[start:end, 1:-1]
        )


def run_test(n, p):
    rng = np.random.default_rng(seed)
    source = mp.RawArray(ctypes.c_double, n * n)
    output = mp.RawArray(ctypes.c_double, n * n)

    a = np.frombuffer(source, dtype=np.float64).reshape(n, n)
    b = np.frombuffer(output, dtype=np.float64).reshape(n, n)

    a[:] = rng.random((n, n))
    b[:] = 0

    chunks = np.array_split(np.arange(1, n - 1), p)
    processes = []

    start_time = time.perf_counter()

    for chunk in chunks:
        if len(chunk) == 0:
            continue

        process = mp.Process(
            target=stencil_worker,
            args=(source, output, n, int(chunk[0]), int(chunk[-1]) + 1, iterations)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError("Process failed")

    elapsed = time.perf_counter() - start_time

    reference = 0.2 * (
        a[:-2, 1:-1]
        + a[2:, 1:-1]
        + a[1:-1, :-2]
        + a[1:-1, 2:]
        + a[1:-1, 1:-1]
    )

    valid = np.allclose(b[1:-1, 1:-1], reference, rtol=1e-12, atol=1e-12)
    checksum = float(np.sum(b[1:-1, 1:-1], dtype=np.float64))

    return elapsed, valid, checksum


def main():
    os.makedirs("results", exist_ok=True)
    results = []
    strong_times = {}

    for p in workers:
        elapsed, valid, checksum = run_test(N, p)

        if not valid:
            raise ValueError("Strong scaling verification failed")

        strong_times[p] = elapsed
        speedup = strong_times[1] / elapsed
        efficiency = speedup / p * 100

        serial_fraction = None
        if p > 1:
            serial_fraction = (1 / speedup - 1 / p) / (1 - 1 / p)

        results.append([
            "strong", p, N, elapsed, speedup,
            efficiency, serial_fraction, checksum, valid
        ])

        print("Strong:", p, round(elapsed, 3), flush=True)

    weak_times = {}

    for p in workers:
        n = math.ceil(N * math.sqrt(p))
        elapsed, valid, checksum = run_test(n, p)

        if not valid:
            raise ValueError("Weak scaling verification failed")

        weak_times[p] = elapsed

        work1 = (N - 2) ** 2 * iterations
        workp = (n - 2) ** 2 * iterations

        scaled_speedup = (
            p * work1 * strong_times[1] / (workp * elapsed)
        )

        results.append([
            "weak", p, n, elapsed, scaled_speedup,
            None, None, checksum, valid
        ])

        print("Weak:", p, n, round(elapsed, 3), flush=True)

    with open("results/scaling_metrics.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "type", "processes", "N", "time_seconds",
            "speedup", "efficiency_percent",
            "serial_fraction", "checksum", "valid"
        ])
        writer.writerows(results)

    flops_per_cell = 5
    bytes_per_cell = 48
    arithmetic_intensity = flops_per_cell / bytes_per_cell

    fractions = [
        (1 / (strong_times[1] / strong_times[p]) - 1 / p) / (1 - 1 / p)
        for p in workers if p > 1
    ]

    average_s = sum(fractions) / len(fractions)

    roofline = {
        "cpu": "Apple M2",
        "N": N,
        "iterations": iterations,
        "flops_per_cell": flops_per_cell,
        "bytes_per_cell_assumed": bytes_per_cell,
        "arithmetic_intensity": arithmetic_intensity,
        "arithmetic_intensity_units": "FLOP/byte",
        "strong_scaling": {
            "serial_fraction_estimate": average_s,
            "amdahl_max_speedup": 1 / average_s if average_s > 0 else None
        },
        "peak_memory_bandwidth_gb_s": None,
        "peak_compute_gflops": None,
        "roofline_regime": "undetermined",
        "note": "Peak compute and memory bandwidth are needed for a verified Roofline classification. DRAM traffic is approximated."
    }

    with open("results/roofline_summary.json", "w") as f:
        json.dump(roofline, f, indent=4)

    print("Results saved")


if __name__ == "__main__":
    mp.freeze_support()
    main()