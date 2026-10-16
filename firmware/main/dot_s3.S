// Скалярное произведение одной строки весов W с nrows строками X на PIE (ESP32-S3).
// a2 = x (выровнен по 16), a3 = w (выровнен по 16), a4 = K/16, a5 = out (int32*),
// a6 = nrows, a7 = шаг между строками x в байтах
    .text
    .align 4
    .global dot_rows_s8
    .type dot_rows_s8, @function
dot_rows_s8:
    entry   a1, 32
    addi    a10, a4, -1
.Lrow:
    mov     a8, a2
    mov     a9, a3
    ee.zero.accx
    ee.vld.128.ip   q0, a8, 16
    ee.vld.128.ip   q1, a9, 16
    loopgtz a10, .Lend
    ee.vmulas.s8.accx.ld.ip  q0, a8, 16, q0, q1
    ee.vld.128.ip   q1, a9, 16
.Lend:
    ee.vmulas.s8.accx q0, q1
    nop
    nop
    rur.accx_0 a11
    s32i    a11, a5, 0
    addi    a5, a5, 4
    add     a2, a2, a7
    addi    a6, a6, -1
    bnez    a6, .Lrow
    retw
    .size dot_rows_s8, . - dot_rows_s8
