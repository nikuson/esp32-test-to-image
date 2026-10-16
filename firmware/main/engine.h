#pragma once
#include <stdint.h>
typedef struct {
    int64_t gemm, quant, ln, attn, softmax, gelu, im2col, dec_w, dec_epi;
    int64_t text, dit, dec, total, macs;
} prof_t;
extern prof_t prof;
int engine_load(void);
int engine_init(void);
void engine_set(int dual, int fast, int pie);
int tokenize(const char *s, int *ids);
int engine_generate(const char *prompt, uint32_t seed, int steps, float cfg, float shift, uint8_t *rgb);
extern int g_stop_after;
