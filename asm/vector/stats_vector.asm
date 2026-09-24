; =============================================================
; stats_vector.asm
; Version VECTORIZADA (AVX2, 8 floats por iteracion) de los
; kernels de computo. Misma ABI que la version escalar.
;
; Antes de compilar/ejecutar en su maquina, confirme soporte AVX2:
;   lscpu | grep avx2
;   cat /proc/cpuinfo | grep avx2
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n -> retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO. Fijense especialmente en:
;   (1) como se calcula cuantos elementos entran en bucles de 8
;       ("and ecx, ~7" redondea n hacia abajo al multiplo de 8),
;   (2) la REDUCCION HORIZONTAL para pasar de 8 sumas parciales
;       (un YMM) a un unico escalar,
;   (3) el BUCLE ESCALAR DE CIERRE para el remanente (n % 8 != 0).
; Reutilicen este mismo patron en compute_stats y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax               ; eax = i = 0
    vxorps  ymm0, ymm0, ymm0       ; ymm0 = acumulador vectorial (8 carriles) = 0

    mov     ecx, esi
    and     ecx, ~7                ; ecx = n redondeado hacia abajo, multiplo de 8
    test    ecx, ecx
    jle     .sum_reduce

.sum_vec_loop:
    cmp     eax, ecx
    jge     .sum_reduce
    vmovups ymm1, [rdi + rax*4]    ; carga 8 floats (unaligned: siempre valido)
    vaddps  ymm0, ymm0, ymm1       ; acumula por carril
    add     eax, 8
    jmp     .sum_vec_loop

.sum_reduce:
    ; --- reduccion horizontal: 8 carriles de ymm0 -> un escalar ---
    vextractf128 xmm2, ymm0, 1     ; xmm2 = mitad alta (carriles 4-7)
    vaddps  xmm0, xmm0, xmm2       ; xmm0 = 4 sumas parciales (carriles 0-3 + 4-7)
    vhaddps xmm0, xmm0, xmm0       ; suma horizontal dentro de 128 bits
    vhaddps xmm0, xmm0, xmm0       ; xmm0[0] = suma total de los 8 carriles originales

.sum_scalar_tail:
    ; --- elementos sobrantes (n % 8), uno a la vez ---
    cmp     eax, esi
    jge     .sum_done
    vmovss  xmm1, [rdi + rax*4]
    vaddss  xmm0, xmm0, xmm1
    inc     eax
    jmp     .sum_scalar_tail

.sum_done:
    vzeroupper                     ; evita penalizacion de transicion AVX/SSE
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
; TODO (estudiante):
;   1) mean = suma(arr) / n (puede llamar a sum_array; recuerde
;      guardar arr/n/mean*/var*/min*/max* en registros callee-saved
;      antes, porque la llamada destruye registros caller-saved).
;   2) Segunda pasada VECTORIZADA para acumular sum((x-mean)^2):
;        - "broadcast" de mean a los 8 carriles con vbroadcastss.
;        - vsubps + vmulps (o vfmadd231ps si quieren ir mas alla)
;          para acumular los cuadrados de las diferencias,
;        - misma reduccion horizontal que en sum_array,
;        - bucle escalar para el remanente (subss/mulss/addss).
;   3) Min/max VECTORIZADOS con vminps/vmaxps a lo largo del bucle
;      principal, reduccion final con vextractf128 + vminps/vmaxps
;      (y shuffles si quieren reducir los 4 restantes a 1), mas
;      bucle escalar de cierre con minss/maxss o comiss.
;   4) Guarde los resultados en [rdx]=mean, [rcx]=var, [r8]=min,
;      [r9]=max. Si n == 0, escriba 0.0 en los cuatro.
;   5) 'vzeroupper' antes de cualquier 'ret' en una funcion que usa
;      registros YMM.
; ---------------------------------------------------------------
compute_stats:
    test    esi, esi
    jz      .stats_empty

    ; Guardar registros callee-saved
    ; para preservar estados
    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8               ; alinear stack a 16 bytes antes de call

    ; Guardar argumentos en registros callee-saved antes de call sum_array.
    mov     rbx, rdi               ; rbx = arr
    mov     r12d, esi              ; r12d = n
    mov     r13, rdx               ; r13 = mean*
    mov     r14, rcx               ; r14 = var*
    mov     r15, r8                ; r15 = min*
    mov     rbp, r9                ; rbp = max*

    ; Preparar argumentos para llamar a sum_array
    mov     rdi, rbx
    mov     esi, r12d
    call    sum_array              ; xmm0 = sum

    ; Calcular mean = sum / n
    vcvtsi2ss xmm4, xmm4, r12d     ; xmm4 = float(n)
    vdivss  xmm0, xmm0, xmm4       ; xmm0 = sum / float(n) = mean
    vmovaps xmm12, xmm0            ; xmm12 = mean, guardar copia
    vbroadcastss ymm6, xmm0        ; ymm6 = xmm0[0] = mean repetido en 8 carriles

    ; Preparar acumuladores vectoriales.
    xor     eax, eax               ; eax = i = 0
    mov     ecx, r12d
    and     ecx, ~7                ; ecx = limite vectorial
    vxorps  ymm7, ymm7, ymm7       ; ymm7 = acc_var vectorial = 0
    vbroadcastss ymm10, [rbx]      ; ymm10 = min vectorial inicial = arr[0]
    vbroadcastss ymm11, [rbx]      ; ymm11 = max vectorial inicial = arr[0]

    test    ecx, ecx                ; lim_vectorial == 0?
    jle     .stats_no_vec_blocks

; Loop vectorial: 8 elementos por iteracion
.stats_vec_loop:
    cmp     eax, ecx
    jge     .stats_reduce_vec
    vmovups ymm8, [rbx + rax*4]    ; ymm8 = arr[i..i+7]
    vminps  ymm10, ymm10, ymm8     ; min por carril
    vmaxps  ymm11, ymm11, ymm8     ; max por carril
    vsubps  ymm9, ymm8, ymm6       ; diff = x - mean
    vmulps  ymm9, ymm9, ymm9       ; diff^2
    vaddps  ymm7, ymm7, ymm9       ; acumula varianza parcial por carril
    add     eax, 8
    jmp     .stats_vec_loop


.stats_reduce_vec:

    ; Reducir acc_var: 8 carriles -> xmm5[0].
    vextractf128 xmm5, ymm7, 1
    vaddps  xmm5, xmm5, xmm7
    vhaddps xmm5, xmm5, xmm5
    vhaddps xmm5, xmm5, xmm5

    ; Reducir min: 8 carriles -> xmm2[0].
    vextractf128 xmm2, ymm10, 1
    vminps  xmm2, xmm2, xmm10
    vpermilps xmm1, xmm2, 0b10110001
    vminps  xmm2, xmm2, xmm1
    vpermilps xmm1, xmm2, 0b01001110
    vminps  xmm2, xmm2, xmm1

    ; Reducir max: 8 carriles -> xmm3[0].
    vextractf128 xmm3, ymm11, 1
    vmaxps  xmm3, xmm3, xmm11
    vpermilps xmm1, xmm3, 0b10110001
    vmaxps  xmm3, xmm3, xmm1
    vpermilps xmm1, xmm3, 0b01001110
    vmaxps  xmm3, xmm3, xmm1
    jmp     .stats_scalar_tail


.stats_no_vec_blocks:
    ; Para n < 8 no hubo bloque vectorial real: iniciar escalares.
    vxorps  xmm5, xmm5, xmm5       ; xmm5 = acc_var escalar = 0
    vmovss  xmm2, [rbx]            ; xmm2 = min = arr[0]
    vmovss  xmm3, [rbx]            ; xmm3 = max = arr[0]

    ; Bucle escalar para el remanente 
.stats_scalar_tail:
    cmp     eax, r12d
    jge     .stats_done
    vmovss  xmm1, [rbx + rax*4]    ; xmm1 = x = arr[i]
    vminss  xmm2, xmm2, xmm1       ; min = min(min, x)
    vmaxss  xmm3, xmm3, xmm1       ; max = max(max, x)
    vsubss  xmm6, xmm1, xmm12      ; xmm6 = x - mean
    vmulss  xmm6, xmm6, xmm6       ; xmm6 = (x - mean)^2
    vaddss  xmm5, xmm5, xmm6       ; acc_var += (x - mean)^2
    inc     eax
    jmp     .stats_scalar_tail







    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    vzeroupper
    ret

.stats_empty:
    vxorps  xmm0, xmm0, xmm0
    vmovss  [rdx], xmm0
    vmovss  [rcx], xmm0
    vmovss  [r8], xmm0
    vmovss  [r9], xmm0
    vzeroupper
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual.
;
; TODO (estudiante):
;   - "Broadcast" mean y stddev a registros YMM con vbroadcastss
;     (guarde antes xmm0/xmm1 en otros registros o en la pila, ya
;     que planea usar xmm0/xmm1 tambien como temporales del bucle).
;   - Bucle vectorial de 8 en 8: vmovups/vmovaps carga, vsubps,
;     vdivps (o vmulps por el reciproco de stddev si quieren
;     optimizar), vmovups/vmovaps guarda.
;   - Bucle escalar de cierre para el remanente (n % 8), igual que
;     en sum_array.
;   - 'vzeroupper' antes del 'ret'.
; ---------------------------------------------------------------
normalize_array:
    ; TODO: implementar
    ret
