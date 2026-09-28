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
"""
import struct
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


def check_normalized_output(input_values, output_path, mean, stddev, tol):
    out_n, out_values = read_input(output_path)
    in_n = len(input_values)

    if out_n != in_n:
        print()
        print("Verificacion de normalize_array:")
        print(f"N de salida esperado={in_n}, obtenido={out_n}  FALLA")
        return False

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
    print()
    print("Verificacion de normalize_array:")
    print(f"{'n':<12}{in_n:>12}")
    print(f"{'max_error':<12}{max_err:>12.6g}")
    if max_idx >= 0:
        print(
            f"{'peor_i':<12}{max_idx:>12}  "
            f"referencia={max_ref:.9g} obtenido={max_val:.9g}"
        )
    print("normalize_array:", "OK" if ok else "FALLA")
    return ok


def main():
    input_path, summary_path, output_path, tol = parse_args(sys.argv)

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
    print(f"{'campo':<10}{'referencia':>15}{'obtenido':>15}{'error rel.':>15}  resultado")
    for name, ref, val in checks:
        err = abs(val - ref) if name == "n" else rel_error(val, ref)
        ok = (err == 0) if name == "n" else (err <= tol)
        all_ok = all_ok and ok
        status = "OK" if ok else "FALLA"
        print(f"{name:<10}{ref:>15.6f}{val:>15.6f}{err:>15.6g}  {status}")

    if output_path is not None:
        all_ok = check_normalized_output(values, output_path, ref_mean, ref_std, tol) and all_ok

    print()
    print("RESULTADO GENERAL:", "PASA" if all_ok else "FALLA")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
