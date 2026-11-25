# Prellenado de Inkling-2-Mini: de dónde sale el tiempo y un GEMM NVFP4 directo

RTX 3090 compartida con el escritorio, caché FP8, modo residente, `IK2_BATCH_PREFILL=1`, prompt de 16 384 tokens. Medido con `bench/ab/run_ab.sh` (PR #29), 3 repeticiones alternadas por sesión.

## Dónde va el tiempo

En `run_prefill_moe`, cada visita a una capa decodifica a FP16 en VRAM cada expert que usan los tokens del chunk (`decode_f16`), en las 48 visitas, y después multiplica con cuBLAS. Para medir ese costo se probó una build que saltea la decodificación (la salida es incorrecta; solo sirve como cota):

| Config | tok/s | Δ |
|---|---:|---:|
| `--chunk 2048` | 2578 | — |
| `--chunk 4096` | 3090 | +19,8 % |
| sin decodificar, chunk 2048 | 3832 | +48,6 % |
| sin decodificar, chunk 4096 | 3933 | +52,5 % |

Un tercio del prellenado es la decodificación. Un chunk mayor la reparte entre más tokens, porque se hace una vez por chunk.

## `IK2_PF_NVFP4=1`: GEMMs agrupados que leen NVFP4

`k::experts_pf_nvfp4` reemplaza la decodificación por expert y el GEMM de cuBLAS. Hay dos versiones; las dos dan exactamente los mismos números.

- **v1** (`IK2_PF_KERNEL=1`): decodifica los pesos a FP16 en memoria compartida, un paso de 64 de K a la vez.
- **v2** (21–25; por defecto la 23): copia los bloques NVFP4 crudos (36 bytes por fila y paso, un cuarto de FP16) a memoria compartida con `cp.async`. Cada hilo decodifica en registros justo sus fragmentos B del `mma.sync`: una lectura de 16 bits da los nibbles bajos (b0) y los altos (b1), que se multiplican por la escala del sub-bloque en FP16. Es exacto, porque E2M1 × UE4M3 entra en FP16.

### Microbenchmark (`tests/ik2_pf_bench.cu`)

128 experts con pesos aleatorios y 32 768 filas (chunk 4096) repartidas de forma despareja, de 11 a 1529 por expert. Se mide gate|up (2304×2304) y down (2304×1152) por separado; mediana de 5.

| Kernel | gate\|up | down | total | Δ |
|---|---:|---:|---:|---:|
| decode + cuBLAS (el actual) | 7,83 ms (44 TFLOPS) | 4,19 ms | 12,02 ms | — |
| v1, tiles de 64 | 6,07 ms | 3,16 ms | 9,24 ms | +30 % |
| v1, tiles de 128 | 5,25 ms | 3,05 ms | 8,30 ms | +45 % |
| v2 21 (128×128, 3 etapas, warps 2×4) | 5,21 ms | 2,77 ms | 7,97 ms | +51 % |
| v2 22 (128×128, 3 etapas, warps 4×2) | 6,11 ms | 3,48 ms | 9,59 ms | +25 % |
| **v2 23 (128×128, 2 etapas, warps 2×4)** | **4,61 ms (75 TFLOPS)** | **2,54 ms** | **7,15 ms** | **+68 %** |
| v2 24 (64×128, 3 etapas) | 6,12 ms | 3,17 ms | 9,29 ms | +29 % |
| v2 25 (128×128, 4 etapas) | 5,59 ms | 2,86 ms | 8,45 ms | +42 % |

Con 2 etapas entran dos bloques por SM (46 KiB de memoria compartida cada uno), y eso pesa más que tener más etapas. Todas las variantes dan diff 0 contra v1.

### En el motor

Al principio, con chunk 4096 el kernel no se usaba. El camino "batch" junta todas las filas de una visita (32 768) en un búfer que a ese tamaño no alcanza, y el motor volvía al camino de a un expert. Ahora, cuando el lote no entra, los experts van en tandas de hasta T filas (los búferes por expert). Cada tanda junta sus filas, corre los dos GEMMs agrupados y suma en `tmp` expert por expert en orden, como antes.

| Prellenado 16K, RTX 3090, FP8 | tok/s (mediana [rango]) | Δ |
|---|---:|---:|
| chunk 2048, decode + cuBLAS | 2504 | — |
| chunk 2048, v1 (batch) | 2982 | +19 % |
| **chunk 4096, decode + cuBLAS** | **3067 [2860–3068]** | — |
| **chunk 4096, v2 23 (tandas)** | **3478 [3477–3482]** | **+13,4 %** |

El techo medido saltándose la decodificación era 3933, así que queda una parte por ganar: los GEMMs de cuBLAS ya eran rápidos y el kernel no los supera por mucho más.

### Corrección

- **`tests/ik2_kernels_test.cu`** (`test_experts_pf_nvfp4`): compara contra la decodificación FP16 y un GEMM en double, en las formas del modelo, con experts de 1 a 130 filas. El peor caso usa el 5 % de la tolerancia.
- **Control de paridad** (`bench/parity`, modos `dump` y `decode`, FP16) con `IK2_PF_NVFP4=1`, en el camino batch y en el de tandas: pasa en los dos, con top-1 98,5 % y KL media 2,7e-3 en `dump`. Sin el kernel daba 97,8 % y 3,1e-3.

Queda apagado por defecto: es opcional hasta que corra más tiempo en uso real.

### El techo era en parte un artefacto (2026-11-25)

La cota "sin decodificar" (+27 % sobre chunk 4096) se midió con una build que da salida incorrecta, y eso cambia el ruteo. Con `IK2_PROF=1` (sincroniza cada fase, prellenado de 16K, chunk 4096) se ve que esa build usa muchos menos experts distintos: 13 213 aciertos de caché frente a 23 651. Hace menos trabajo que el modelo real, así que no es un techo alcanzable.

| Prellenado 16K, chunk 4096, `IK2_PROF=1` | total | experts (GPU) |
|---|---:|---:|
| decode + cuBLAS | 6,10 s | 2,99 s |
| `IK2_PF_NVFP4=1` | 4,86 s | 2,26 s |
| sin decodificar (ruteo distinto: no comparable) | 4,55 s | 1,96 s |

En el microbenchmark, cuBLAS sin decodificar (pesos ya en FP16) tarda 8,34 ms por visita y el kernel v2 7,15 ms: **el kernel ya es más rápido que cuBLAS sin ningún costo de decodificación**, a ~90 % del pico de los tensor cores en gate|up. Lo que queda en los experts está fuera del GEMM:
- las 4 capas Q8_0, que siguen con decode + cuBLAS;
- la reunión de filas;
- el SwiGLU, que lee y escribe `gu32` en F32 (~0,3 GB por visita);
- la suma ponderada en `tmp`.
