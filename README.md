# Normalizador estadistico: NASM escalar y AVX2

Este proyecto implementa un normalizador estadistico para arreglos de
`float32`. Incluye dos versiones funcionalmente equivalentes de los kernels:

- una version escalar, que procesa un elemento por iteracion;
- una version vectorial AVX2, que procesa ocho elementos por iteracion y
  maneja el remanente con un bucle escalar.

El programa calcula suma, media, varianza poblacional, desviacion estandar,
minimo y maximo. Luego genera el arreglo normalizado:

```text
out[i] = (in[i] - mean) / stddev
```

Cuando `stddev` es cero, la normalizacion copia el arreglo de entrada para
evitar una division por cero.

## Estructura

```text
.
├── asm/
│   ├── scalar/stats_scalar.asm   # Kernels escalares
│   └── vector/stats_vector.asm   # Kernels AVX2
├── include/stats.h               # Interfaz C/NASM
├── src/driver.c                  # E/S, memoria alineada y medicion
├── tools/
│   ├── gen_input.py              # Generacion de entradas binarias
│   ├── verify_reference.py       # Verificacion contra referencia Python
│   └── run_benchmark.py          # Comparacion escalar vs. vectorial
├── data/                         # Entradas y salidas generadas
└── Makefile
```

## Requisitos

- Linux x86-64 con soporte AVX2 para ejecutar la version vectorial.
- NASM, GCC, Make y Python 3.

Se puede confirmar el soporte AVX2 con:

```bash
lscpu | grep avx2
```

## Compilacion

```bash
make
```

Se generan los ejecutables:

```text
bin/norm_scalar
bin/norm_vector
```

## Verificacion funcional

```bash
make check
```

Este comando compila ambas versiones y ejecuta una matriz de casos de prueba:
arreglo vacio, tamanos pequenos, tamanos no multiplos de ocho, bloques
vectoriales completos, datos constantes y valores extremos.

## Benchmark

```bash
make benchmark
```

El benchmark genera entradas de `N = 10^3`, `10^5`, `10^6` y `5 x 10^7`;
ejecuta ambas versiones con 30 repeticiones por defecto; y muestra tiempo
promedio, desviacion estandar y speedup. Los resultados se guardan en
`data/benchmark_results.csv`.

Para cambiar la cantidad de repeticiones:

```bash
make benchmark REPS=50
```

## Ejecucion manual

Primero se genera una entrada:

```bash
python3 tools/gen_input.py 1000 data/input.dat random
```

Luego se ejecuta cualquiera de las versiones:

```bash
./bin/norm_scalar data/input.dat data/output_scalar.dat 30
./bin/norm_vector data/input.dat data/output_vector.dat 30
```

El tercer argumento indica la cantidad de repeticiones del kernel. Cada
ejecucion escribe los estadisticos y los tiempos en un archivo
`*.stats.txt` asociado a la salida.

Para comprobar una ejecucion particular contra la referencia:

```bash
python3 tools/verify_reference.py data/input.dat data/output_vector.dat.stats.txt data/output_vector.dat
```

## Limpieza

```bash
make clean
```
