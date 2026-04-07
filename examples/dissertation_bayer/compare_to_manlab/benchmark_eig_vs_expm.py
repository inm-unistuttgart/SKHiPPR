import argparse
import time
from tqdm import tqdm

import numpy as np
from scipy.linalg import expm


def time_call(func, matrix):
    start = time.perf_counter()
    func(matrix)
    stop = time.perf_counter()
    return stop - start


def summarize(values):
    values = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Quick benchmark: compare np.linalg.eig and scipy.linalg.expm "
            "on random square matrices."
        )
    )
    parser.add_argument(
        "--size", type=int, default=1000, help="Matrix size n for n x n"
    )
    parser.add_argument(
        "--num-matrices",
        type=int,
        default=3,
        help="Number of random matrices in the benchmark set",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="How many times to benchmark each matrix",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    matrices = [
        rng.standard_normal((args.size, args.size)) for _ in range(args.num_matrices)
    ]

    eig_times = []
    expm_times = []

    for matrix in tqdm(matrices):
        for _ in range(args.repeats):
            eig_times.append(time_call(np.linalg.eig, matrix))
            expm_times.append(time_call(expm, matrix))

    eig_stats = summarize(eig_times)
    expm_stats = summarize(expm_times)

    print("Benchmark settings")
    print(f"  matrix size: {args.size} x {args.size}")
    print(f"  number of matrices: {args.num_matrices}")
    print(f"  repeats per matrix: {args.repeats}")
    print(f"  total timed calls per method: {len(eig_times)}")
    print()

    print("np.linalg.eig timing [s]")
    print(f"  mean:   {eig_stats['mean']:.6f}")
    print(f"  median: {eig_stats['median']:.6f}")
    print(f"  min:    {eig_stats['min']:.6f}")
    print(f"  max:    {eig_stats['max']:.6f}")
    print()

    print("scipy.linalg.expm timing [s]")
    print(f"  mean:   {expm_stats['mean']:.6f}")
    print(f"  median: {expm_stats['median']:.6f}")
    print(f"  min:    {expm_stats['min']:.6f}")
    print(f"  max:    {expm_stats['max']:.6f}")
    print()

    if expm_stats["median"] > 0.0:
        print(
            f"speedup (eig/expm median): {eig_stats['median'] / expm_stats['median']:.3f}x"
        )
    else:
        print("speedup (eig/expm median): n/a")


if __name__ == "__main__":
    main()
