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
; Recorre bloques alineados de 8 floats con AVX2, reduce horizontalmente
; las 8 sumas parciales y termina con un bucle escalar para el remanente.
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
    vmovaps ymm1, [rdi + rax*4]    ; carga alineada: base 32 B + avance de 32 B
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
; Implementacion:
;   1) Calcula mean = sum_array(arr, n) / n.
;   2) Recorre bloques de 8 floats para acumular sum((x - mean)^2)
;      en carriles vectoriales y actualizar min/max por carril.
;   3) Reduce horizontalmente varianza parcial, min y max a escalares.
;   4) Procesa el remanente con instrucciones escalares.
;   5) Guarda [rdx]=mean, [rcx]=var, [r8]=min, [r9]=max.
;      Si n == 0, escribe 0.0 en los cuatro resultados.
; ---------------------------------------------------------------
compute_stats:
    test    esi, esi
    jz      .stats_empty

    ; Guardar registros callee-saved usados por la funcion.
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

    ; Preparar argumentos para llamar a sum_array.
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
    vmovaps ymm8, [rbx + rax*4]    ; ymm8 = arr[i..i+7], direccion alineada a 32 B
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

    ; Bucle escalar para el remanente.
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

    ; Guardar resultados en las direcciones de puntero
.stats_done:
    vdivss  xmm5, xmm5, xmm4       ; var = acc_var / float(n)
    vmovss  [r13], xmm12           ; *mean = mean
    vmovss  [r14], xmm5            ; *var = var
    vmovss  [r15], xmm2            ; *min = min
    vmovss  [rbp], xmm3            ; *max = max

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
; Implementacion:
;   - Si stddev == 0.0, copia in -> out para evitar division por cero.
;   - Si stddev != 0.0, replica mean/stddev en registros YMM y normaliza
;     bloques alineados de 8 floats con vmovaps, vsubps y vdivps.
;   - Los elementos restantes se procesan con instrucciones escalares.
; ---------------------------------------------------------------
normalize_array:
    xor     eax, eax               ; eax = i = 0
    mov     ecx, edx
    and     ecx, ~7                ; ecx = limite vectorial

    vxorps  xmm2, xmm2, xmm2       ; xmm2 = 0.0
    vucomiss xmm1, xmm2            ; stddev == 0.0?
    je      .norm_copy_vec_loop

    ; Caso stddev != 0.0: out[i] = (in[i] - mean) / stddev
    vmovaps xmm8, xmm0             ; xmm8[0] = mean escalar
    vmovaps xmm9, xmm1             ; xmm9[0] = stddev escalar
    vbroadcastss ymm10, xmm0       ; ymm10 = [mean, ..., mean] en 8 carriles
    vbroadcastss ymm11, xmm1       ; ymm11 = [stddev, ..., stddev] en 8 carriles

    ; Bucle vectorial
.norm_vec_loop:
    cmp     eax, ecx                ; i >= lim_vect?
    jge     .norm_scalar_tail
    vmovaps ymm12, [rdi + rax*4]   ; ymm12 = in[i..i+7]
    vsubps  ymm12, ymm12, ymm10    ; ymm12 = in - mean
    vdivps  ymm12, ymm12, ymm11    ; ymm12 = (in - mean) / stddev
    vmovaps [rsi + rax*4], ymm12   ; out[i..i+7] = resultado
    add     eax, 8
    jmp     .norm_vec_loop

    ; Bucle escalar de remanente
.norm_scalar_tail:
    cmp     eax, edx
    jge     .norm_done
    vmovss  xmm12, [rdi + rax*4]   ; xmm12 = in[i]
    vsubss  xmm12, xmm12, xmm8     ; xmm12 = in[i] - mean
    vdivss  xmm12, xmm12, xmm9     ; xmm12 = (in[i] - mean) / stddev
    vmovss  [rsi + rax*4], xmm12   ; out[i] = xmm12
    inc     eax
    jmp     .norm_scalar_tail

    ; Caso stddev == 0.0: copiar in -> out en vectorial
.norm_copy_vec_loop:
    cmp     eax, ecx                    ; i >= lim_vect?
    jge     .norm_copy_scalar_tail      ; Copia remanente escalar
    vmovaps ymm12, [rdi + rax*4]        ; ymm12 = in[i..i+7]
    vmovaps [rsi + rax*4], ymm12        ; out[i..i+7] = ymm12
    add     eax, 8
    jmp     .norm_copy_vec_loop

    ; Copia escalar del remanente
.norm_copy_scalar_tail:
    cmp     eax, edx                    ; i >= n?
    jge     .norm_done
    vmovss  xmm12, [rdi + rax*4]        ; xmm12 = in[i]
    vmovss  [rsi + rax*4], xmm12        ; out[i] = xmm12
    inc     eax
    jmp     .norm_copy_scalar_tail

.norm_done:
    vzeroupper
    ret
