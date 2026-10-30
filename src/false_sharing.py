import multiprocessing as mp
import ctypes
import time
import csv
import os

K = 5000000
workers = [1, 2, 4, 8]
modes = ["unpadded", "padded", "privatized"]

def worker(arr, p, iterations, mode):
    if mode == "unpadded":
        for i in range(iterations):
            arr[p] += 1

    elif mode == "padded":
        for i in range(iterations):
            arr[p * 8] += 1

    else:
        total = 0
        for i in range(iterations):
            total += 1
        arr[p] = total


def run_test(p, mode):
    size = p * 8 if mode == "padded" else p
    arr = mp.RawArray(ctypes.c_int64, size)

    processes = []
    start = time.perf_counter()

    for i in range(p):
        process = mp.Process(target=worker, args=(arr, i, K, mode))
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        if process.exitcode != 0:
            raise RuntimeError("Worker failed")

    elapsed = time.perf_counter() - start

    if mode == "padded":
        total = sum(arr[i * 8] for i in range(p))
    else:
        total = sum(arr[i] for i in range(p))

    if total != p * K:
        raise ValueError("Incorrect counter total")

    return elapsed, total


def main():
    os.makedirs("results", exist_ok=True)
    results = []

    for mode in modes:
        times = {}

        for p in workers:
            elapsed, total = run_test(p, mode)
            times[p] = elapsed

            speedup = times[1] / elapsed
            karp_flatt = None

            if p > 1:
                karp_flatt = (p * elapsed / times[1] - 1) / (p - 1)

            results.append([
                mode, p, K, elapsed, speedup, karp_flatt, total
            ])

            print(mode, p, round(elapsed, 3), total, flush=True)

    with open("results/false_sharing_karp_flatt.tsv", "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "mode", "processes", "iterations",
            "time_seconds", "speedup", "karp_flatt", "total"
        ])
        writer.writerows(results)

    print("Results saved")


if __name__ == "__main__":
    mp.freeze_support()
    main()