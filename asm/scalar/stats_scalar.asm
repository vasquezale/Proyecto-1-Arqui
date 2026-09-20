; =============================================================
; stats_scalar.asm
; Version ESCALAR (referencia) de los kernels de computo.
;
; Convencion de llamada: System V AMD64 ABI
;   enteros/punteros: rdi, rsi, rdx, rcx, r8, r9
;   flotantes:        xmm0, xmm1, xmm2, ...
;   retorno float:    xmm0
;   callee-saved:     rbx, rbp, r12-r15 (si los usa, debe preservarlos)
; =============================================================

    global sum_array
    global compute_stats
    global normalize_array

    section .text

; ---------------------------------------------------------------
; float sum_array(const float *arr, int n)
;   rdi = arr, esi = n
;   retorna la suma en xmm0
;
; IMPLEMENTADA COMO EJEMPLO: estudien este patron (recorrido,
; acumulador, condicion de salida) antes de escribir compute_stats
; y normalize_array.
; ---------------------------------------------------------------
sum_array:
    xor     eax, eax           ; eax = i = 0
    xorps   xmm0, xmm0         ; xmm0 = acumulador = 0.0

.sum_loop:
    cmp     eax, esi
    jge     .sum_done
    movss   xmm1, [rdi + rax*4]
    addss   xmm0, xmm1
    inc     eax
    jmp     .sum_loop

.sum_done:
    ret

; ---------------------------------------------------------------
; void compute_stats(const float *arr, int n,
;                     float *mean, float *var, float *min, float *max)
;   rdi = arr, esi = n, rdx = mean*, rcx = var*, r8 = min*, r9 = max*
;
;   var = varianza POBLACIONAL = sum((x - mean)^2) / n
;   Caso borde: si n == 0, escriba 0.0 en mean/var/min/max.
;
; TODO (estudiante):
;   1) Calcular mean = suma(arr) / n. Puede reutilizar sum_array con
;      'call sum_array', pero recuerde que eso destruye los
;      registros caller-saved (rax, rcx, rdx, rsi, rdi, r8-r11):
;      guarde arr/n/mean*/var*/min*/max* en registros callee-saved
;      (rbx, r12-r15) ANTES de llamar.
;   2) Recorrer el arreglo una segunda vez para acumular
;      sum((x - mean)^2) y obtener var = esa suma / n.
;   3) Recorrer el arreglo (puede combinarlo con el paso 1) llevando
;      min y max con comiss + saltos condicionales (ja/jb, etc.)
;      o con las instrucciones minss/maxss.
;   4) Guardar los resultados en las direcciones recibidas por
;      puntero: [rdx]=mean, [rcx]=var, [r8]=min, [r9]=max.
;   5) No olvide restaurar los registros callee-saved en el epilogo.
; ---------------------------------------------------------------
compute_stats:

    test esi, esi
    jz      .stats_empty

    push    rbx
    push    rbp
    push    r12
    push    r13
    push    r14
    push    r15
    sub     rsp, 8               ; alinear stack a 16 bytes antes de call

    ; Guardar argumentos en registros para preservarlos con la llamada a sum_array
    mov     rbx, rdi            ; rdx = arr
    mov     r12d, esi           ; r12 = n,     r12d usa la parte baja de r12 para operaciones de 32 bits      
    mov     r13, rdx            ; r13 = mean*
    mov     r14, rcx            ; r14 = var*
    mov     r15, r8             ; r15 = min*
    mov     rbp, r9             ; rbp = max*

    ; Calcular mean = sum(arr, n) / n
    mov     rdi, rbx            ; sum_array espera arr en rdi por ABI
    mov     esi, r12d           ; sum_array espera n en esi 
    call    sum_array           ; xmm0 = sum
    cvtsi2ss xmm4, r12d         ; xmm4 = float(n)
    divss   xmm0, xmm4          ; xmm0 = mean

    ; Inicializar Loop para calcular var, min y max
    xor     eax, eax            ; i = 0
    movss   xmm2, [rbx]         ; xmm2 = arr[0] inicializar min
    movss   xmm3, [rbx]         ; xmm3 = arr[0] inicializar max
    xorps   xmm5, xmm5          ; xmm5 = acc_var = 0.0

    ; Loop para calcular var, min y max del arreglo
.stats-loop:
    cmp     eax, r12d           ; i >= n? Termina el Loop
    jge     .stats-done

    movss   xmm1, [rbx + rax*4]   ; xmm1 = arr[i]
    minss   xmm2, xmm1            ; min = min(min, arr[i])
    maxss   xmm3, xmm1            ; max = max(max, arr[i])

    movss   xmm6, xmm1            ; xmm6 = arr[i]
    subss   xmm6, xmm0            ; xmm6 = arr[i] - mean
    mulss   xmm6, xmm6            ; xmm6 = (arr[i] - mean)^2
    addss   xmm5, xmm6            ; acc_var += (arr[i] - mean)^2

    inc     eax
    jmp     .stats-loop

    ; Fin del Loop, calcular var final y guarda resultados
.stats-done:
    divss   xmm5, xmm4          ; xmm5 = var = acc_var / float(n)
    movss   [r13], xmm0         ; *mean = mean
    movss   [r14], xmm5         ; *var = var
    movss   [r15], xmm2         ; *min = min
    movss   [rbp], xmm3         ; *max = max

    add     rsp, 8
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rbp
    pop     rbx
    ret

    ; Caso borde: si n == 0, escriba 0.0 en mean/var/min/max.
.stats_empty:
    xorps   xmm0, xmm0
    movss   [rdx], xmm0
    movss   [rcx], xmm0
    movss   [r8], xmm0
    movss   [r9], xmm0
    ret

; ---------------------------------------------------------------
; void normalize_array(const float *in, float *out, int n,
;                       float mean, float stddev)
;   rdi = in, rsi = out, edx = n, xmm0 = mean, xmm1 = stddev
;
;   out[i] = (in[i] - mean) / stddev
;   Caso borde: si stddev == 0.0, copie in[i] en out[i] tal cual
;   (evite division por cero).
;
; TODO (estudiante): implementar el bucle escalar.
; Sugerencia: guarde mean (xmm0) y stddev (xmm1) en registros que no
; se sobrescriban dentro del bucle (por ejemplo xmm8/xmm9, que en
; System V no se usan para pasar argumentos), o vuelva a cargarlos
; en cada iteracion desde una copia guardada en la pila.
; ---------------------------------------------------------------
normalize_array:
    ; TODO: implementar
    ret
