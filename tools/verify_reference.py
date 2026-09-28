#!/usr/bin/env python3
"""
Calcula estadisticos de referencia (en Python puro, sin SIMD) para un
archivo input.dat y los compara contra el resumen que el driver en C
escribe en '<output>.stats.txt'.

Opcionalmente compara tambien el archivo binario de salida normalizada
contra la referencia esperada de normalize_array.

Uso:
    python3 verify_reference.py <input.dat> <output.stats.txt> [tolerancia]
    python3 verify_reference.py <input.dat> <output.stats.txt> <output.dat> [tolerancia]
    python3 verify_reference.py --matrix [tolerancia]
"""
import os
import random
import struct
import subprocess
import sys
import math


def read_input(path):
    with open(path, "rb") as f:
        header = f.read(4)
        if len(header) != 4:
            raise ValueError(f"archivo invalido: {path} no contiene N")
        n = struct.unpack("<i", header)[0]
        if n < 0:
            raise ValueError(f"archivo invalido: N negativo ({n}) en {path}")
        data = f.read(4 * n)
        if len(data) != 4 * n:
            raise ValueError(f"archivo truncado: {path}")
        values = list(struct.unpack(f"<{n}f", data)) if n > 0 else []
    return n, values


def read_summary(path):
    result = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, val = line.split("=", 1)
            result[key] = float(val)
    return result


def reference_stats(values):
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    total = sum(values)
    mean = total / n
    var = sum((x - mean) ** 2 for x in values) / n
    stddev = math.sqrt(var)
    return total, mean, var, stddev, min(values), max(values)


def gen_values(n, mode, seed):
    rng = random.Random(seed)
    if mode == "random":
        return [rng.uniform(-100.0, 100.0) for _ in range(n)]
    if mode == "constant":
        return [5.0 for _ in range(n)]
    if mode == "edge":
        base = [-1e6, 1e6, 0.0, -0.0001, 0.0001, -1.0, 1.0]
        return [base[i % len(base)] for i in range(n)]
    raise ValueError(f"modo desconocido: {mode}")


def write_dat(path, values):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(values)))
        if values:
            f.write(struct.pack(f"<{len(values)}f", *values))


def rel_error(a, b):
    if abs(b) < 1e-12:
        return abs(a - b)
    return abs(a - b) / abs(b)


def parse_args(argv):
    if len(argv) < 3:
        print(
            f"Uso: {argv[0]} <input.dat> <output.stats.txt> [tolerancia]\n"
            f"     {argv[0]} <input.dat> <output.stats.txt> <output.dat> [tolerancia]"
        )
        sys.exit(1)

    input_path = argv[1]
    summary_path = argv[2]
    output_path = None
    tol = 1e-4

    if len(argv) >= 4:
        try:
            tol = float(argv[3])
        except ValueError:
            output_path = argv[3]

    if len(argv) >= 5:
        if output_path is None:
            print("Error: la tolerancia debe ser el ultimo argumento.")
            sys.exit(1)
        tol = float(argv[4])

    if len(argv) > 5:
        print("Error: demasiados argumentos.")
        sys.exit(1)

    return input_path, summary_path, output_path, tol


def normalized_result(input_values, output_path, mean, stddev, tol):
    out_n, out_values = read_input(output_path)
    in_n = len(input_values)

    if out_n != in_n:
        return False, in_n, 0.0, -1, 0.0, 0.0, out_n

    max_err = 0.0
    max_idx = -1
    max_ref = 0.0
    max_val = 0.0

    for i, x in enumerate(input_values):
        ref = x if abs(stddev) < 1e-12 else (x - mean) / stddev
        val = out_values[i]
        err = rel_error(val, ref)
        if err > max_err:
            max_err = err
            max_idx = i
            max_ref = ref
            max_val = val

    ok = max_err <= tol
    return ok, in_n, max_err, max_idx, max_ref, max_val, out_n


def check_normalized_output(input_values, output_path, mean, stddev, tol):
    ok, in_n, max_err, max_idx, max_ref, max_val, out_n = normalized_result(
        input_values, output_path, mean, stddev, tol
    )
    print()
    print("Verificacion de normalize_array:")
    if out_n != in_n:
        print(f"N de salida esperado={in_n}, obtenido={out_n}  FALLA")
        return False
    print(f"{'n':<12}{in_n:>12}")
    print(f"{'max_error':<12}{max_err:>12.6g}")
    if max_idx >= 0:
        print(
            f"{'peor_i':<12}{max_idx:>12}  "
            f"referencia={max_ref:.9g} obtenido={max_val:.9g}"
        )
    print("normalize_array:", "OK" if ok else "FALLA")
    return ok


def verify_reference(input_path, summary_path, output_path, tol, quiet=False):
    n, values = read_input(input_path)
    ref_sum, ref_mean, ref_var, ref_std, ref_min, ref_max = reference_stats(values)
    got = read_summary(summary_path)

    checks = [
        ("n", float(n), got.get("n", float("nan"))),
        ("sum", ref_sum, got.get("sum", float("nan"))),
        ("mean", ref_mean, got.get("mean", float("nan"))),
        ("var", ref_var, got.get("var", float("nan"))),
        ("stddev", ref_std, got.get("stddev", float("nan"))),
        ("min", ref_min, got.get("min", float("nan"))),
        ("max", ref_max, got.get("max", float("nan"))),
    ]

    all_ok = True
    if not quiet:
        print(f"{'campo':<10}{'referencia':>15}{'obtenido':>15}{'error rel.':>15}  resultado")
    for name, ref, val in checks:
        err = abs(val - ref) if name == "n" else rel_error(val, ref)
        ok = (err == 0) if name == "n" else (err <= tol)
        all_ok = all_ok and ok
        if not quiet:
            status = "OK" if ok else "FALLA"
            print(f"{name:<10}{ref:>15.6f}{val:>15.6f}{err:>15.6g}  {status}")

    if output_path is not None:
        if quiet:
            norm_ok = normalized_result(values, output_path, ref_mean, ref_std, tol)[0]
        else:
            norm_ok = check_normalized_output(values, output_path, ref_mean, ref_std, tol)
        all_ok = norm_ok and all_ok

    if not quiet:
        print()
        print("RESULTADO GENERAL:", "PASA" if all_ok else "FALLA")
    return all_ok


def run_command(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def run_matrix(tol):
    cases = [
        ("empty", 0, "random", 1000),
        ("n1", 1, "random", 1001),
        ("n7", 7, "random", 1007),
        ("n8", 8, "random", 1008),
        ("n15", 15, "random", 1015),
        ("n16", 16, "random", 1016),
        ("n1000", 1000, "random", 2000),
        ("constant16", 16, "constant", 0),
        ("edge15", 15, "edge", 0),
        ("edge1001", 1001, "edge", 0),
    ]
    versions = [
        ("scalar", "bin/norm_scalar"),
        ("vector", "bin/norm_vector"),
    ]

    all_ok = True
    results = []
    print("Generando entradas y verificando matriz de correctud...")
    for case_name, n, mode, seed in cases:
        input_path = f"data/check_{case_name}.dat"
        write_dat(input_path, gen_values(n, mode, seed))

        row = {"case": case_name}
        for version, binary in versions:
            output_path = f"data/check_output_{version}_{case_name}.dat"
            stats_path = f"{output_path}.stats.txt"
            proc = run_command([f"./{binary}", input_path, output_path, "1"])
            if proc.returncode != 0:
                row[version] = "FALLA"
                all_ok = False
                print()
                print(f"Fallo ejecutando {binary} para caso {case_name}:")
                if proc.stdout:
                    print(proc.stdout)
                if proc.stderr:
                    print(proc.stderr)
                continue

            ok = verify_reference(input_path, stats_path, output_path, tol, quiet=True)
            row[version] = "PASA" if ok else "FALLA"
            all_ok = all_ok and ok
            if not ok:
                print()
                print(f"Detalle de falla: caso={case_name}, version={version}")
                verify_reference(input_path, stats_path, output_path, tol, quiet=False)
        results.append(row)

    print()
    print(f"{'caso':<14}{'scalar':>10}{'vector':>10}")
    for row in results:
        print(f"{row['case']:<14}{row.get('scalar', 'FALLA'):>10}{row.get('vector', 'FALLA'):>10}")
    print()
    print("RESULTADO GENERAL:", "PASA" if all_ok else "FALLA")
    return all_ok


def main():
    if len(sys.argv) >= 2 and sys.argv[1] == "--matrix":
        tol = float(sys.argv[2]) if len(sys.argv) >= 3 else 1e-4
        sys.exit(0 if run_matrix(tol) else 1)

    input_path, summary_path, output_path, tol = parse_args(sys.argv)
    sys.exit(0 if verify_reference(input_path, summary_path, output_path, tol) else 1)


if __name__ == "__main__":
    main()
