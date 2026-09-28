#!/usr/bin/env python3
"""
Ejecuta el benchmark escalar vs vectorial para los tamanos exigidos.

Uso:
    python3 tools/run_benchmark.py [repeticiones]

Genera entradas reproducibles en data/bench_<N>.dat, ejecuta ambas
versiones con el mismo dataset, lee los tiempos del resumen .stats.txt
y guarda data/benchmark_results.csv.
"""
import csv
import os
import random
import struct
import subprocess
import sys


BENCHMARK_SIZES = [1_000, 100_000, 1_000_000, 50_000_000]
DEFAULT_REPS = 30
DATA_DIR = "data"
RESULTS_CSV = os.path.join(DATA_DIR, "benchmark_results.csv")
WRITE_CHUNK = 1_000_000


def write_input(path, n, seed):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = random.Random(seed)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", n))
        remaining = n
        while remaining > 0:
            count = min(remaining, WRITE_CHUNK)
            values = [rng.uniform(-100.0, 100.0) for _ in range(count)]
            f.write(struct.pack(f"<{count}f", *values))
            remaining -= count


def read_summary(path):
    result = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            result[key] = float(value)
    return result


def run_command(cmd):
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print(f"Fallo ejecutando: {' '.join(cmd)}")
        if proc.stdout:
            print(proc.stdout)
        if proc.stderr:
            print(proc.stderr)
        sys.exit(proc.returncode)
    return proc


def run_kernel(binary, input_path, output_path, reps):
    run_command([binary, input_path, output_path, str(reps)])
    summary = read_summary(f"{output_path}.stats.txt")
    mean_ms = summary.get("kernel_ms_mean", summary.get("kernel_ms"))
    stddev_ms = summary.get("kernel_ms_stddev", 0.0)
    if mean_ms is None:
        print(f"Falta kernel_ms_mean/kernel_ms en {output_path}.stats.txt")
        sys.exit(1)
    return mean_ms, stddev_ms


def main():
    reps = int(sys.argv[1]) if len(sys.argv) >= 2 else DEFAULT_REPS
    if reps < 1:
        print("Error: repeticiones debe ser >= 1")
        sys.exit(1)

    rows = []
    print(f"Benchmark escalar vs vectorial, reps={reps}")
    print("Generando datasets y ejecutando kernels...")

    for n in BENCHMARK_SIZES:
        input_path = os.path.join(DATA_DIR, f"bench_{n}.dat")
        seed = 12345 + n
        print(f"  N={n}: generando {input_path}")
        write_input(input_path, n, seed)

        scalar_output = os.path.join(DATA_DIR, f"bench_output_scalar_{n}.dat")
        vector_output = os.path.join(DATA_DIR, f"bench_output_vector_{n}.dat")

        print(f"  N={n}: ejecutando escalar")
        scalar_mean, scalar_stddev = run_kernel(
            "./bin/norm_scalar", input_path, scalar_output, reps
        )

        print(f"  N={n}: ejecutando vectorial")
        vector_mean, vector_stddev = run_kernel(
            "./bin/norm_vector", input_path, vector_output, reps
        )

        speedup = scalar_mean / vector_mean if vector_mean != 0.0 else float("inf")
        rows.append({
            "N": n,
            "reps": reps,
            "scalar_mean_ms": scalar_mean,
            "scalar_stddev_ms": scalar_stddev,
            "vector_mean_ms": vector_mean,
            "vector_stddev_ms": vector_stddev,
            "speedup": speedup,
        })

    with open(RESULTS_CSV, "w", newline="") as f:
        fieldnames = [
            "N",
            "reps",
            "scalar_mean_ms",
            "scalar_stddev_ms",
            "vector_mean_ms",
            "vector_stddev_ms",
            "speedup",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"{'N':>12}{'scalar_ms':>14}{'scalar_sd':>14}"
          f"{'vector_ms':>14}{'vector_sd':>14}{'speedup':>12}")
    for row in rows:
        print(
            f"{row['N']:>12}"
            f"{row['scalar_mean_ms']:>14.6f}"
            f"{row['scalar_stddev_ms']:>14.6f}"
            f"{row['vector_mean_ms']:>14.6f}"
            f"{row['vector_stddev_ms']:>14.6f}"
            f"{row['speedup']:>12.4f}x"
        )

    print()
    print(f"Resultados guardados en {RESULTS_CSV}")


if __name__ == "__main__":
    main()
