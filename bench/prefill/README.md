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

## `IK2_PF_NVFP4=1`: un GEMM agrupado que lee NVFP4

`k::experts_pf_nvfp4` procesa todos los experts de una visita en un lanzamiento por sentido (gate|up y down). Los pesos se decodifican en memoria compartida, un paso de 64 de K a la vez, con la aritmética exacta de `decode_f16`. El producto va por `mma.sync` m16n8k16 con sumas F32 y doble búfer (`cp.async` para X, registros para los pesos).

- **Corrección:** `tests/ik2_kernels_test.cu` (`test_experts_pf_nvfp4`) lo compara contra la decodificación FP16 y un GEMM en double, en las formas del modelo, con experts de 1 a 130 filas y tiles de 64 y de 128. El error queda en el 5 % de la tolerancia.
- **Paridad del modelo** (`bench/parity`, modo `dump`, `IK2_BATCH_PREFILL=1`): pasa, con top-1 98,5 % frente a 97,8 % y KL media 2,7e-3 frente a 3,1e-3 del camino actual.

| Prellenado 16K | tok/s | Δ |
|---|---:|---:|
| chunk 2048, decode + cuBLAS | 2504 | — |
| chunk 2048, `IK2_PF_NVFP4=1` | 2982 | **+19 %** |
| chunk 4096, decode + cuBLAS | 2929 / 3055 (2 sesiones) | — |
| chunk 4096, `IK2_PF_NVFP4=1` (tiles de 128) | 2944 / 2948 | +0,5 % / −3,5 % (ruido) |
| chunk 4096, tiles de 64 | 3023 | −1,1 % (ruido) |

Con chunk 4096 empata. Se ahorra la decodificación, pero aprovecha los tensor cores peor que cuBLAS: las tiles de 128 filas decodifican la mitad y no cambian nada. Por eso queda apagado por defecto. Sirve donde `--chunk 4096` no entra en VRAM. Para acercarse al techo (+27 % sobre chunk 4096) haría falta afinarlo: más etapas, más bloques por SM, tiles más anchas.
