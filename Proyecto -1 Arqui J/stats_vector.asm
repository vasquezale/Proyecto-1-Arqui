global sum_array

section .text

sum_array:
    xor     eax, eax
    vxorps  ymm0, ymm0, ymm0
    mov     ecx, esi
    and     ecx, ~7
    test    ecx, ecx
    jle     .sum_reduce

.sum_vec_loop:
    cmp     eax, ecx
    jge     .sum_reduce
    vmovups ymm1, [rdi + rax*4]
    vaddps  ymm0, ymm0, ymm1
    add     eax, 8
    jmp     .sum_vec_loop

.sum_reduce:
    vextractf128 xmm2, ymm0, 1
    vaddps  xmm0, xmm0, xmm2
    vhaddps xmm0, xmm0, xmm0
    vhaddps xmm0, xmm0, xmm0

.sum_scalar_tail:
    cmp     eax, esi
    jge     .sum_done
    vmovss  xmm1, [rdi + rax*4]
    vaddss  xmm0, xmm0, xmm1
    inc     eax
    jmp .sum_scalar_tail

.sum_done:
    vzeroupper
    ret