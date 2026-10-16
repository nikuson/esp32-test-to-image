// Text-to-image на ESP32-S3: текстовый энкодер + DiT (rectified flow) + conv-декодер.
// Вся тяжёлая математика — одно int8 GEMM-ядро на PIE (dot_s3.S), раскиданное на 2 ядра.
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "esp_partition.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "engine.h"

void dot_rows_s8(const int8_t *x, const int8_t *w, int k16, int32_t *out, int nrows, int xstride);

// ---------------------------------------------------------------- профилировщик
prof_t prof;
#define TIC() int64_t _t0 = esp_timer_get_time()
#define TOC(f) prof.f += esp_timer_get_time() - _t0

// ---------------------------------------------------------------- модель из flash
typedef struct { int O, K, Kp; const int8_t *w; const float *s, *b; } ql_t;
typedef struct { const float *n1w, *n1b, *n2w, *n2b; ql_t qkv, o, ma, mb; } tblock_t;
typedef struct { const float *table; ql_t qkv, o, a, b; } dblock_t;
typedef struct { int Ci, Co, relu_in, up, kind; ql_t q; } conv_t;

static struct {
    int V, D, DEPTH, HEADS, MLP, TD, TDEPTH, ZC, ZH, P, MAXLEN, STEM;
    float lat_mean, lat_std, cfg;
    int nconv, nscales; const float *scales;
    const char *vocab; const float *emb, *tpos;
    tblock_t tb[4]; ql_t tout;
    const float *null, *pos; ql_t patch, t1, t2, ada;
    dblock_t db[12]; ql_t fada, out;
    conv_t conv[16];
} M;

static const uint8_t *cur;
static const void *take(size_t n) { cur = (const uint8_t *)(((uintptr_t)cur + 15) & ~15); const void *p = cur; cur += n; return p; }
static int rd_i32(void) { int v; memcpy(&v, cur, 4); cur += 4; return v; }
static float rd_f32(void) { float v; memcpy(&v, cur, 4); cur += 4; return v; }
static ql_t rd_ql(void) {
    ql_t q; take(0); q.O = rd_i32(); q.K = rd_i32(); q.Kp = rd_i32(); rd_i32();
    q.w = take(q.O * q.Kp); q.s = take(q.O * 4); q.b = take(q.O * 4); return q;
}

int engine_load(void) {
    const esp_partition_t *part = esp_partition_find_first(ESP_PARTITION_TYPE_DATA, 0x40, "model");
    if (!part) { printf("no model partition\n"); return -1; }
    const void *map; esp_partition_mmap_handle_t h;
    if (esp_partition_mmap(part, 0, part->size, ESP_PARTITION_MMAP_DATA, &map, &h) != ESP_OK) { printf("mmap failed\n"); return -1; }
    cur = map;
    if (memcmp(cur, "TTI1", 4)) { printf("bad magic\n"); return -1; }
    cur += 4;
    M.V = rd_i32(); M.D = rd_i32(); M.DEPTH = rd_i32(); M.HEADS = rd_i32(); M.MLP = rd_i32();
    M.TD = rd_i32(); M.TDEPTH = rd_i32(); M.ZC = rd_i32(); M.ZH = rd_i32(); M.P = rd_i32();
    M.MAXLEN = rd_i32(); M.STEM = rd_i32();
    M.lat_mean = rd_f32(); M.lat_std = rd_f32(); M.cfg = rd_f32();
    M.nconv = rd_i32(); M.nscales = rd_i32(); M.scales = (const float *)cur; cur += 4 * M.nscales;
    M.vocab = take(M.V * 16);
    M.emb = take(M.V * M.TD * 4); M.tpos = take(M.MAXLEN * M.TD * 4);
    for (int i = 0; i < M.TDEPTH; i++) {
        tblock_t *b = &M.tb[i];
        b->n1w = take(M.TD * 4); b->n1b = take(M.TD * 4); b->qkv = rd_ql(); b->o = rd_ql();
        b->n2w = take(M.TD * 4); b->n2b = take(M.TD * 4); b->ma = rd_ql(); b->mb = rd_ql();
    }
    M.tout = rd_ql();
    M.null = take(M.D * 4); M.patch = rd_ql(); M.pos = take((M.ZH / M.P) * (M.ZH / M.P) * M.D * 4);
    M.t1 = rd_ql(); M.t2 = rd_ql(); M.ada = rd_ql();
    for (int i = 0; i < M.DEPTH; i++) {
        dblock_t *b = &M.db[i];
        b->table = take(6 * M.D * 4); b->qkv = rd_ql(); b->o = rd_ql(); b->a = rd_ql(); b->b = rd_ql();
    }
    M.fada = rd_ql(); M.out = rd_ql();
    for (int i = 0; i < M.nconv; i++) {
        conv_t *c = &M.conv[i]; take(0);
        c->Ci = rd_i32(); c->Co = rd_i32(); c->relu_in = rd_i32(); c->up = rd_i32(); c->kind = rd_i32();
        c->q = rd_ql();
    }
    take(0);
    if (memcmp(cur, "END!", 4)) { printf("bad tail\n"); return -1; }
    printf("model: V=%d D=%d depth=%d heads=%d, %d convs, %u bytes\n",
           M.V, M.D, M.DEPTH, M.HEADS, M.nconv, (unsigned)(cur + 4 - (const uint8_t *)map));
    return 0;
}

// ---------------------------------------------------------------- int8 GEMM на двух ядрах
// Y[t][o] (ldy) = acc(t,o) * sx[t] * sw[o] + b[o]         (EPI_STORE)
// Y[t][o]      += gate[t>=split][o] * (acc*sx*sw + b[o])   (EPI_GATED_ADD)
enum { EPI_STORE, EPI_GATED_ADD };
typedef struct {
    const int8_t *x; int ldx; const float *sx; int T;
    const int8_t *w; int ldw; int k16; const float *sw, *b;
    float *y; int ldy; int epi; const float *g0, *g1; int split;
} gemm_t;

#define MAXT 128
static int use_pie = 1;
// эталонное ядро на чистом C — для сравнения в статье
static void dot_rows_c(const int8_t *x, const int8_t *w, int k16, int32_t *out, int nrows, int xstride) {
    const int K = k16 * 16;
    for (int t = 0; t < nrows; t++, x += xstride) {
        int32_t s = 0;
        for (int k = 0; k < K; k++) s += x[k] * w[k];
        out[t] = s;
    }
}
#define DOT(...) (use_pie ? dot_rows_s8(__VA_ARGS__) : dot_rows_c(__VA_ARGS__))
static void gemm_range(const gemm_t *g, int o0, int o1) {
    int32_t acc[MAXT];
    for (int o = o0; o < o1; o++) {
        DOT(g->x, g->w + o * g->ldw, g->k16, acc, g->T, g->ldx);
        const float so = g->sw[o], bo = g->b ? g->b[o] : 0.f;
        float *y = g->y + o;
        if (g->epi == EPI_STORE) {
            for (int t = 0; t < g->T; t++) y[t * g->ldy] = acc[t] * (g->sx[t] * so) + bo;
        } else {
            const float ga = g->g0[o], gb = g->g1 ? g->g1[o] : ga;
            for (int t = 0; t < g->T; t++)
                y[t * g->ldy] += (t < g->split ? ga : gb) * (acc[t] * (g->sx[t] * so) + bo);
        }
    }
}

// ---- par(): одна и та же функция на двух ядрах, каждое берёт свою половину диапазона.
// Синхронизация — task notifications (дешевле семафоров).
typedef void (*par_fn)(void *arg, int part, int nparts);
static par_fn s_fn; static void *s_arg;
static TaskHandle_t s_worker, s_main;
static int use_dual = 1;
#define SPLIT(n, p, np) int lo = (n) * (p) / (np), hi = (n) * ((p) + 1) / (np)
static void worker(void *arg) {
    for (;;) { ulTaskNotifyTake(pdTRUE, portMAX_DELAY); s_fn(s_arg, 1, 2); xTaskNotifyGive(s_main); }
}
static void par(par_fn fn, void *arg, int n) {
    if (use_dual && n >= 2) {
        s_fn = fn; s_arg = arg; s_main = xTaskGetCurrentTaskHandle();
        xTaskNotifyGive(s_worker);
        fn(arg, 0, 2);
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    } else fn(arg, 0, 1);
}

typedef struct { const gemm_t *g; int O; } gemm_job_t;
static void gemm_part(void *a, int p, int np) { gemm_job_t *j = a; SPLIT(j->O, p, np); gemm_range(j->g, lo, hi); }
static void gemm(const gemm_t *g, int O) {
    TIC();
    gemm_job_t j = {g, O};
    par(gemm_part, &j, O >= 8 ? 2 : 1);
    TOC(gemm);
    prof.macs += (int64_t)g->T * g->k16 * 16 * O;
}

// Linear: x int8 [T][Kp], y float [T][ldy]
static void linear(const ql_t *q, const int8_t *x, const float *sx, int T, float *y, int ldy) {
    gemm_t g = {x, q->Kp, sx, T, q->w, q->Kp, q->Kp / 16, q->s, q->b, y, ldy, EPI_STORE, 0, 0, 0};
    gemm(&g, q->O);
}

// ---------------------------------------------------------------- поэлементные операции
// Округление к ближайшему чётному (как np.round) без libm: магическое число 1.5*2^23.
// lrintf() на Xtensa — это вызов библиотечной функции, а нам нужно ~2 инструкции.
static inline int32_t rne(float x) {
    union { float f; int32_t i; } u = {x + 12582912.0f};
    return u.i - 0x4B400000;
}

static int fast_math = 1;



// exp для softmax: 2^x через разбиение на целую часть (в экспоненту float) и полином 3-й степени
static inline float fexp(float x) {
    if (!fast_math) return expf(x);
    if (x < -80.f) return 0.f;
    float t = x * 1.442695041f;
    int32_t ii = (int32_t)t; if (t < (float)ii) ii--;   // floor без libm
    float f = t - (float)ii;
    float p = 1.f + f * (0.6951786f + f * (0.2261557f + f * 0.0781693f));
    union { float f; int32_t i; } u = {p};
    u.i += ii << 23;
    return u.f;
}
// GELU(tanh) через таблицу на [-6, 6] с линейной интерполяцией
#define GL_N 1024
static float gelu_lut[GL_N + 2];
static float gelu_slope[GL_N + 1];
static inline float gelu_exact(float x) { return 0.5f * x * (1.f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))); }
static void gelu_init(void) {
    for (int i = 0; i <= GL_N + 1; i++) {
        float x = -6.f + 12.f * i / GL_N;
        gelu_lut[i] = gelu_exact(x);
    }
    for (int i = 0; i <= GL_N; i++) gelu_slope[i] = gelu_lut[i + 1] - gelu_lut[i];
}
// табличная GELU: одна проверка диапазона, индекс, одно FMA (наклоны посчитаны заранее)
static inline float gelu(float x) {
    float p = (x + 6.f) * (GL_N / 12.f);
    if (p <= 0.f) return 0.f;
    if (p >= (float)GL_N) return x;
    int i = (int)p;
    return gelu_lut[i] + (p - (float)i) * gelu_slope[i];
}
static inline float silu(float x) { return x / (1.f + expf(-x)); }

// Квантование строк float -> int8 с масштабом на строку; mode 1 = сначала GELU на месте
typedef struct { float *x; int ldx, T, K; int8_t *q; int ldq; float *s; int mode; } qr_t;
static void qr_part(void *a, int p, int np) {
    qr_t *j = a; SPLIT(j->T, p, np);
    for (int t = lo; t < hi; t++) {
        float *r = j->x + t * j->ldx; float m = 0;
        if (j->mode == 1) {
            if (fast_math) for (int k = 0; k < j->K; k++) r[k] = gelu(r[k]);
            else for (int k = 0; k < j->K; k++) r[k] = gelu_exact(r[k]);
        }
        for (int k = 0; k < j->K; k++) { float a = fabsf(r[k]); if (a > m) m = a; }
        float sc = m / 127.f; if (sc < 1e-12f) sc = 1e-12f;
        j->s[t] = sc; float inv = 1.f / sc;
        int8_t *o = j->q + t * j->ldq;
        for (int k = 0; k < j->K; k++) o[k] = (int8_t)rne(r[k] * inv);
        for (int k = j->K; k < j->ldq; k++) o[k] = 0;
    }
}
static void quant_rows_m(const float *x, int ldx, int T, int K, int8_t *q, int ldq, float *s, int mode) {
    TIC();
    qr_t j = {(float *)x, ldx, T, K, q, ldq, s, mode};
    par(qr_part, &j, T >= 8 ? 2 : 1);
    if (mode) { TOC(gelu); } else { TOC(quant); }
}
#define quant_rows(x, ldx, T, K, q, ldq, s) quant_rows_m(x, ldx, T, K, q, ldq, s, 0)

// LayerNorm (+ adaLN-модуляция или аффинные веса) + сразу квантование строки в int8
typedef struct { const float *x; int ldx, T, K; const float *mul0, *add0, *mul1, *add1; int split, onepl; float eps;
                 int8_t *q; int ldq; float *s; } ln_t;
static void ln_part(void *a, int p, int np) {
    ln_t *j = a; SPLIT(j->T, p, np);
    const int K = j->K;
    float tmp[256];
    for (int t = lo; t < hi; t++) {
        const float *r = j->x + t * j->ldx;
        float mu = 0; for (int k = 0; k < K; k++) mu += r[k]; mu /= K;
        float var = 0; for (int k = 0; k < K; k++) { float d = r[k] - mu; var += d * d; } var /= K;
        float inv = 1.f / sqrtf(var + j->eps);
        const float *mul = t < j->split ? j->mul0 : j->mul1, *add = t < j->split ? j->add0 : j->add1;
        float m = 0;
        for (int k = 0; k < K; k++) {
            float v = (r[k] - mu) * inv * (j->onepl ? 1.f + mul[k] : mul[k]) + add[k];
            tmp[k] = v; float a = fabsf(v); if (a > m) m = a;
        }
        float sc = m / 127.f; if (sc < 1e-12f) sc = 1e-12f;
        j->s[t] = sc; float iv = 1.f / sc;
        int8_t *o = j->q + t * j->ldq;
        for (int k = 0; k < K; k++) o[k] = (int8_t)rne(tmp[k] * iv);
        for (int k = K; k < j->ldq; k++) o[k] = 0;
    }
}
static void ln_mod_quant(const float *x, int ldx, int T, int K, const float *mul0, const float *add0,
                         const float *mul1, const float *add1, int split, int onepl, float eps,
                         int8_t *q, int ldq, float *s) {
    TIC();
    ln_t j = {x, ldx, T, K, mul0, add0, mul1, add1, split, onepl, eps, q, ldq, s};
    par(ln_part, &j, T >= 8 ? 2 : 1);
    TOC(ln);
}

// ---------------------------------------------------------------- int8 внимание одной головы
// q,k,v: float, строки с шагом ld; out float [T][ldo]; hd = 32
static int8_t *a_q, *a_k, *a_vt, *a_p; static float *a_sq, *a_sk, *a_svt, *a_sp, *a_sc;
typedef struct { const float *v; int ld, T, Tp; const uint8_t *keep; float *out; int ldo; } at_t;
static void at_scores(void *a, int p, int np) {        // S[i][j] = q_i·k_j, j делим между ядрами
    at_t *j = a; SPLIT(j->T, p, np);
    int32_t acc[MAXT]; const float isq = 0.17677669529f;  // 1/sqrt(32)
    for (int c = lo; c < hi; c++) {
        DOT(a_q, a_k + c * 32, 2, acc, j->T, 32);
        float f = a_sk[c] * isq;
        for (int i = 0; i < j->T; i++) a_sc[i * j->Tp + c] = (j->keep && !j->keep[c]) ? -1e9f : acc[i] * a_sq[i] * f;
    }
}
static void at_softmax(void *a, int p, int np) {       // softmax + квантование P по строкам
    at_t *j = a; SPLIT(j->T, p, np);
    for (int i = lo; i < hi; i++) {
        float *r = a_sc + i * j->Tp, mx = -1e30f, sum = 0;
        for (int c = 0; c < j->T; c++) if (r[c] > mx) mx = r[c];
        for (int c = 0; c < j->T; c++) { r[c] = fexp(r[c] - mx); sum += r[c]; }
        // max(p) = 1/sum, поэтому масштаб квантования известен сразу
        // p = r/sum, max(p) = 1/sum  =>  q = round(p / (max/127)) = round(r * 127)
        a_sp[i] = 1.f / (sum * 127.f);
        int8_t *o = a_p + i * j->Tp;
        for (int c = 0; c < j->T; c++) o[c] = (int8_t)rne(r[c] * 127.f);
        for (int c = j->T; c < j->Tp; c++) o[c] = 0;
    }
}
static void at_out(void *a, int p, int np) {           // V^T по каналам и out = P·V
    at_t *j = a; SPLIT(32, p, np);
    int32_t acc[MAXT];
    for (int c = lo; c < hi; c++) {
        float m = 0;
        for (int t = 0; t < j->T; t++) { float a = fabsf(j->v[t * j->ld + c]); if (a > m) m = a; }
        float sc = m / 127.f; if (sc < 1e-12f) sc = 1e-12f; float iv = 1.f / sc;
        int8_t *o = a_vt + c * j->Tp;
        for (int t = 0; t < j->T; t++) o[t] = (int8_t)rne(j->v[t * j->ld + c] * iv);
        for (int t = j->T; t < j->Tp; t++) o[t] = 0;
        DOT(a_p, o, j->Tp / 16, acc, j->T, j->Tp);
        for (int i = 0; i < j->T; i++) j->out[i * j->ldo + c] = acc[i] * a_sp[i] * sc;
    }
}
static void attn_head(const float *q, const float *k, const float *v, int ld, int T, const uint8_t *keep,
                      float *out, int ldo) {
    const int Tp = (T + 15) & ~15;
    quant_rows(q, ld, T, 32, a_q, 32, a_sq);
    quant_rows(k, ld, T, 32, a_k, 32, a_sk);
    at_t j = {v, ld, T, Tp, keep, out, ldo};
    int n = T >= 8 ? 2 : 1;
    { TIC(); par(at_scores, &j, n); TOC(attn); }
    { TIC(); par(at_softmax, &j, n); TOC(softmax); }
    { TIC(); par(at_out, &j, n); TOC(attn); }
}

// ---------------------------------------------------------------- буферы
static float *X, *ATT, *QKV, *HF, *tmpf;
static int8_t *HQ, *HQ2; static float *SX, *SX2;
static uint8_t *ACT_A, *ACT_B;   // декодер (PSRAM)
static int8_t *COLS; static float *YF; static int8_t *WBUF;

#define IA(n) heap_caps_aligned_alloc(16, (n), MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)
// Одна арена во внутренней SRAM: буферы разных фаз (текст / DiT / декодер) живут по одним адресам.
//   X   [128][192] f32  96K  — residual stream DiT;  в декодере: COLS + YF + WBUF
//   R   72K               — ATT [64][192] f32 + QKV [64][96] f32  |  HF [128][96] f32 (MLP)
//   HQ  [128][192] i8 24K, HQ2 [128][96] i8 12K, scratch внимания ~27K
int engine_init(void) {
    xTaskCreatePinnedToCore(worker, "core1", 4096, NULL, configMAX_PRIORITIES - 2, &s_worker, 1);
    gelu_init();
    const int T = 128, D = 192;
    size_t sz = T * D * 4 + 72 * 1024 + T * D + T * 96 + 4 * 1024 + 64 * 32 * 2 + 32 * 64 + 64 * 64 + 64 * 64 * 4 + 12 * 1024;
    printf("largest internal block %u\n", (unsigned)heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL));
    // арену режем на три куска — куча фрагментирована, одного блока на 250К нет
    uint8_t *a0 = IA(T * D * 4), *a1 = IA(64 * D * 4), *a = IA(sz - T * D * 4 - 72 * 1024), *a2 = IA(64 * 96 * 4);
    if (!a0 || !a1 || !a || !a2) { printf("arena alloc %u failed, free %u\n", (unsigned)sz, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL)); return -1; }
    X = (float *)a0;
    ATT = (float *)a1; QKV = (float *)a2; HF = (float *)a1;   // HF [128][96] == ATT [64][192] по размеру
    HQ = (int8_t *)a; a += T * D; HQ2 = (int8_t *)a; a += T * 96;
    SX = (float *)a; a += 512; SX2 = (float *)a; a += 512; a_sq = (float *)a; a += 512; a_sk = (float *)a; a += 512;
    a_svt = (float *)a; a += 512; a_sp = (float *)a; a += 512; a += 1024;
    a_q = (int8_t *)a; a += 64 * 32; a_k = (int8_t *)a; a += 64 * 32; a_vt = (int8_t *)a; a += 32 * 64;
    a_p = (int8_t *)a; a += 64 * 64; a_sc = (float *)a; a += 64 * 64 * 4; tmpf = (float *)a;
    COLS = (int8_t *)X; YF = (float *)((uint8_t *)X + 64 * 576); WBUF = (int8_t *)((uint8_t *)YF + 64 * 64 * 4);
    ACT_A = heap_caps_aligned_alloc(16, 64 * 64 * 64, MALLOC_CAP_SPIRAM);
    ACT_B = heap_caps_aligned_alloc(16, 64 * 64 * 64, MALLOC_CAP_SPIRAM);
    if (!ACT_A || !ACT_B) { printf("psram alloc failed\n"); return -1; }
    printf("arena %u bytes, free internal after alloc: %u\n", (unsigned)sz, (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL));
    return 0;
}

void engine_set(int dual, int fast, int pie) { use_dual = dual; fast_math = fast; use_pie = pie; }

// ---------------------------------------------------------------- токенизатор
// нижний регистр, ё->е, слова [а-яa-z]+, обрезка до STEM букв, поиск в словаре
int tokenize(const char *s, int *ids) {
    int n = 0; uint32_t word[32]; int wl = 0;
    const uint8_t *p = (const uint8_t *)s;
    for (;;) {
        uint32_t c = 0; int len = 0;
        if (*p == 0) c = 0;
        else if (*p < 0x80) { c = *p; len = 1; }
        else if ((*p & 0xE0) == 0xC0 && p[1]) { c = ((p[0] & 0x1F) << 6) | (p[1] & 0x3F); len = 2; }
        else { c = '?'; len = 1; }
        if (c >= 'A' && c <= 'Z') c += 32;
        if (c >= 0x410 && c <= 0x42F) c += 0x20;
        if (c == 0x401 || c == 0x451) c = 0x435;
        int letter = (c >= 'a' && c <= 'z') || (c >= 0x430 && c <= 0x44F);
        if (letter) { if (wl < 32) word[wl++] = c; }
        else if (wl) {
            char st[16] = {0}; int bl = 0;
            for (int i = 0; i < wl && i < M.STEM; i++) {
                if (word[i] < 0x80) st[bl++] = word[i];
                else { st[bl++] = 0xC0 | (word[i] >> 6); st[bl++] = 0x80 | (word[i] & 0x3F); }
            }
            int id = 1;
            for (int v = 0; v < M.V; v++) if (!memcmp(M.vocab + v * 16, st, 16)) { id = v; break; }
            if (n < M.MAXLEN) ids[n++] = id;
            wl = 0;
        }
        if (c == 0) break;
        p += len;
    }
    return n;
}

// ---------------------------------------------------------------- текстовый энкодер
static void text_encode(const int *ids, int n, float *out) {
    const int TD = M.TD, H = 4, hd = 32;
    float *x = QKV;                 // [n][TD]
    float *qkv = X;                 // [n][3TD]
    float *hid = X + 12 * 3 * TD;   // [n][2TD]
    for (int t = 0; t < n; t++)
        for (int k = 0; k < TD; k++) x[t * TD + k] = M.emb[ids[t] * TD + k] + M.tpos[t * TD + k];
    for (int l = 0; l < M.TDEPTH; l++) {
        tblock_t *b = &M.tb[l];
        ln_mod_quant(x, TD, n, TD, b->n1w, b->n1b, b->n1w, b->n1b, n, 0, 1e-5f, HQ, TD, SX);
        linear(&b->qkv, HQ, SX, n, qkv, 3 * TD);   // строки qkv переставлены: [голова][q|k|v][32]
        float *att = X + 8192;
        for (int h = 0; h < H; h++) {
            const float *base = qkv + h * 3 * hd;
            attn_head(base, base + hd, base + 2 * hd, 3 * TD, n, NULL, att + h * hd, TD);
        }
        quant_rows(att, TD, n, TD, HQ, TD, SX);
        float *o = hid;
        linear(&b->o, HQ, SX, n, o, TD);
        for (int i = 0; i < n * TD; i++) x[i] += o[i];
        ln_mod_quant(x, TD, n, TD, b->n2w, b->n2b, b->n2w, b->n2b, n, 0, 1e-5f, HQ, TD, SX);
        linear(&b->ma, HQ, SX, n, hid, 2 * TD);
        quant_rows_m(hid, 2 * TD, n, 2 * TD, HQ2, 2 * TD, SX2, 1);
        linear(&b->mb, HQ2, SX2, n, o, TD);
        for (int i = 0; i < n * TD; i++) x[i] += o[i];
    }
    float pooled[256] = {0};
    for (int t = 0; t < n; t++) for (int k = 0; k < TD; k++) pooled[k] += x[t * TD + k] / n;
    quant_rows(pooled, TD, 1, TD, HQ, TD, SX);
    linear(&M.tout, HQ, SX, 1, out, M.D);
}

// ---------------------------------------------------------------- DiT: один шаг для 2 ветвей CFG разом
// z: [ZC][ZH][ZH]; c: [2][D] (условие текста / пустое); v: [2][ZC][ZH][ZH]
static void dit_forward(const float *z, float t, const float *ctext, float *v) {
    const int D = M.D, N = (M.ZH / M.P) * (M.ZH / M.P), T = 2 * N, P = M.P, G = M.ZH / M.P, hd = D / M.HEADS;
    float *c = tmpf, *mod = tmpf + 2 * D;   // c [2][D], mod [2][6][D]
    // эмбеддинг времени
    float te[256] = {0};
    for (int i = 0; i < D / 2; i++) {
        float a = t * 1000.f * expf(-logf(10000.f) * i / (D / 2));
        te[i] = cosf(a); te[i + D / 2] = sinf(a);
    }
    float h1[256] = {0}, h2[256] = {0};
    quant_rows(te, D, 1, D, HQ, D, SX); linear(&M.t1, HQ, SX, 1, h1, D);
    for (int i = 0; i < D; i++) h1[i] = silu(h1[i]);
    quant_rows(h1, D, 1, D, HQ, D, SX); linear(&M.t2, HQ, SX, 1, h2, D);
    for (int b = 0; b < 2; b++) for (int i = 0; i < D; i++) c[b * D + i] = silu(ctext[b * D + i] + h2[i]);
    quant_rows(c, D, 2, D, HQ, D, SX);
    linear(&M.ada, HQ, SX, 2, mod, 6 * D);
    // патчи -> токены (обе ветви начинают с одного и того же z)
    float pv[16];
    for (int b = 0; b < 2; b++)
        for (int gy = 0; gy < G; gy++) for (int gx = 0; gx < G; gx++) {
            int n = 0;
            for (int ch = 0; ch < M.ZC; ch++) for (int py = 0; py < P; py++) for (int px = 0; px < P; px++)
                pv[n++] = z[(ch * M.ZH + gy * P + py) * M.ZH + gx * P + px];
            quant_rows(pv, 16, 1, 16, HQ + (b * N + gy * G + gx) * 16, 16, SX + b * N + gy * G + gx);
        }
    linear(&M.patch, HQ, SX, T, X, D);
    for (int t2 = 0; t2 < T; t2++) for (int k = 0; k < D; k++) X[t2 * D + k] += M.pos[(t2 % N) * D + k];

    for (int l = 0; l < M.DEPTH; l++) {
        dblock_t *bl = &M.db[l];
        static float m[2][6][256];
        for (int b = 0; b < 2; b++) for (int j = 0; j < 6; j++) for (int k = 0; k < D; k++)
            m[b][j][k] = mod[(b * 6 + j) * D + k] + bl->table[j * D + k];
        // --- attention
        ln_mod_quant(X, D, T, D, m[0][1], m[0][0], m[1][1], m[1][0], N, 1, 1e-6f, HQ, D, SX);
        for (int b = 0; b < 2; b++) {   // ветви CFG по очереди: экономим SRAM на ATT/QKV
            for (int h = 0; h < M.HEADS; h++) {
                gemm_t g = {HQ + b * N * D, D, SX + b * N, N, bl->qkv.w + h * 3 * hd * D, D, D / 16, bl->qkv.s + h * 3 * hd,
                            bl->qkv.b + h * 3 * hd, QKV, 3 * hd, EPI_STORE, 0, 0, 0};
                gemm(&g, 3 * hd);
                attn_head(QKV, QKV + hd, QKV + 2 * hd, 3 * hd, N, NULL, ATT + h * hd, D);
            }
            quant_rows(ATT, D, N, D, HQ2, D, SX2);
            gemm_t g = {HQ2, D, SX2, N, bl->o.w, D, D / 16, bl->o.s, bl->o.b, X + b * N * D, D, EPI_GATED_ADD, m[b][2], NULL, N};
            gemm(&g, D);
        }
        // --- MLP, скрытый слой кусками по 96 (группированное квантование по K)
        ln_mod_quant(X, D, T, D, m[0][4], m[0][3], m[1][4], m[1][3], N, 1, 1e-6f, HQ, D, SX);
        const int Hd = D * M.MLP;
        for (int ch = 0; ch < Hd; ch += 96) {
            gemm_t g = {HQ, D, SX, T, bl->a.w + ch * D, D, D / 16, bl->a.s + ch, bl->a.b + ch, HF, 96, EPI_STORE, 0, 0, 0};
            gemm(&g, 96);
            quant_rows_m(HF, 96, T, 96, HQ2, 96, SX2, 1);   // GELU + квантование за один проход
            gemm_t g2 = {HQ2, 96, SX2, T, bl->b.w + ch, Hd, 96 / 16, bl->b.s, ch == 0 ? bl->b.b : NULL,
                         X, D, EPI_GATED_ADD, m[0][5], m[1][5], N};
            gemm(&g2, D);
        }
    }
    // финальный слой
    static float fmb[2 * 2 * 256];
    float *fm0 = fmb, *fm1 = fmb + 2 * D;   // строки результата идут с шагом 2D
    quant_rows(c, D, 2, D, HQ, D, SX);
    linear(&M.fada, HQ, SX, 2, fmb, 2 * D);
    ln_mod_quant(X, D, T, D, fm0 + D, fm0, fm1 + D, fm1, N, 1, 1e-6f, HQ, D, SX);
    float *o = QKV; // [T][16]
    linear(&M.out, HQ, SX, T, o, 16);
    for (int b = 0; b < 2; b++)
        for (int gy = 0; gy < G; gy++) for (int gx = 0; gx < G; gx++) {
            const float *r = o + (b * N + gy * G + gx) * 16; int n = 0;
            for (int ch = 0; ch < M.ZC; ch++) for (int py = 0; py < P; py++) for (int px = 0; px < P; px++)
                v[((b * M.ZC + ch) * M.ZH + gy * P + py) * M.ZH + gx * P + px] = r[n++];
        }
}

// ---------------------------------------------------------------- декодер
// Активации int8 HWC в PSRAM, статические масштабы. Апсемплинг «бесплатный» — внутри im2col.
static void conv_layer(const conv_t *cv, const int8_t *in, int H, int W, float sin, int8_t *out, float sout,
                       int8_t *resid, float sres, uint8_t *rgb) {
    const int up = cv->up, Ho = H << up, Wo = W << up, Ci = cv->Ci, Co = cv->Co, Kp = cv->q.Kp;
    TIC();
    memcpy(WBUF, cv->q.w, Co * Kp);   // веса слоя -> SRAM, чтобы не мучить кэш flash
    TOC(dec_w);
    float sx[64]; for (int i = 0; i < 64; i++) sx[i] = sin;
    int8_t *LB = (int8_t *)QKV;   // 3 входные строки в SRAM (QKV в декодере свободен)
    const int rowb = W * Ci;
    for (int y = 0; y < Ho; y++) {
        { TIC();
        // строчный буфер: каждую нужную строку читаем из PSRAM один раз, ReLU — при копировании
        for (int dy = 0; dy < 3; dy++) {
            int yy = y + dy - 1;
            int8_t *dst = LB + dy * rowb;
            if (yy < 0 || yy >= Ho) { memset(dst, 0, rowb); continue; }
            const int8_t *src = in + (yy >> up) * rowb;
            if (cv->relu_in) for (int k = 0; k < rowb; k++) dst[k] = src[k] > 0 ? src[k] : 0;
            else memcpy(dst, src, rowb);
        }
        for (int x = 0; x < Wo; x++) {
            int8_t *col = COLS + x * Kp; int n = 0;
            for (int dy = 0; dy < 3; dy++) for (int dx = -1; dx <= 1; dx++) {
                int xx = x + dx;
                if (xx < 0 || xx >= Wo) memset(col + n, 0, Ci);
                else memcpy(col + n, LB + dy * rowb + (xx >> up) * Ci, Ci);
                n += Ci;
            }
            memset(col + n, 0, Kp - n);
        }
        TOC(im2col); }
        gemm_t g = {COLS, Kp, sx, Wo, WBUF, Kp, Kp / 16, cv->q.s, cv->q.b, YF, Co, EPI_STORE, 0, 0, 0};
        gemm(&g, Co);
        TIC();
        if (rgb) {
            for (int x = 0; x < Wo; x++) for (int k = 0; k < 3; k++) {
                float v = (YF[x * Co + k] + 1.f) * 127.5f;
                rgb[(y * Wo + x) * 3 + k] = v < 0 ? 0 : v > 255 ? 255 : (uint8_t)rne(v);
            }
        } else {
            float is = 1.f / sout;
            int8_t *o = out + y * Wo * Co;
            for (int i = 0; i < Wo * Co; i++) {
                float v = YF[i];
                if (resid) v += resid[y * Wo * Co + i] * sres;
                if (cv->kind == 1 && v < 0) v = 0;
                int q = rne(v * is); o[i] = q > 127 ? 127 : q < -127 ? -127 : q;
            }
        }
        TOC(dec_epi);
    }
}

static void decode(const float *z, uint8_t *rgb) {
    const float *sc = M.scales; int si = 0;
    int H = M.ZH, W = M.ZH, C = M.ZC;
    int8_t *x = (int8_t *)ACT_A, *h = (int8_t *)ACT_B;
    float s_in = sc[si++], s_h = 0;
    for (int yy = 0; yy < H; yy++) for (int xx = 0; xx < W; xx++) for (int c = 0; c < C; c++) {
        float v = (z[(c * H + yy) * W + xx] * M.lat_std + M.lat_mean) / s_in;
        int q = rne(v); x[(yy * W + xx) * C + c] = q > 127 ? 127 : q < -127 ? -127 : q;
    }
    for (int i = 0; i < M.nconv; i++) {
        const conv_t *cv = &M.conv[i];
        int Ho = H << cv->up, Wo = W << cv->up;
        if (cv->kind == 0) {        // обычная свёртка: x -> h, затем меняем местами
            float so = sc[si++];
            conv_layer(cv, x, H, W, s_in, h, so, NULL, 0, NULL);
            int8_t *t = x; x = h; h = t; s_in = so;
        } else if (cv->kind == 1) { // res: h = relu(conv1(relu x))
            s_h = sc[si++];
            conv_layer(cv, x, H, W, s_in, h, s_h, NULL, 0, NULL);
        } else if (cv->kind == 2) { // res: x = x + conv2(h) (на месте)
            float so = sc[si++];
            conv_layer(cv, h, H, W, s_h, x, so, x, s_in, NULL);
            s_in = so;
        } else {
            conv_layer(cv, x, H, W, s_in, NULL, 1, NULL, 0, rgb);
        }
        H = Ho; W = Wo; C = cv->Co;
    }
}

// ---------------------------------------------------------------- ГПСЧ (как в export.py)
static uint32_t rng_x;
static uint32_t u32(void) { uint32_t x = rng_x; x ^= x << 13; x ^= x >> 17; x ^= x << 5; return rng_x = x; }
static void gauss(float *o, int n) {
    for (int i = 0; i < n; i += 2) {
        float u1 = (u32() >> 8) / 16777216.0f + 1e-7f, u2 = (u32() >> 8) / 16777216.0f;
        float r = sqrtf(-2.f * logf(u1));
        o[i] = r * cosf(6.2831853f * u2);
        if (i + 1 < n) o[i + 1] = r * sinf(6.2831853f * u2);
    }
}

// ---------------------------------------------------------------- генерация
int g_stop_after = -1;   // >=0: остановиться после k шагов и декодировать «недорисованный» латент
int engine_generate(const char *prompt, uint32_t seed, int steps, float cfg, float shift, uint8_t *rgb) {
    memset(&prof, 0, sizeof prof);
    int64_t t0 = esp_timer_get_time();
    int ids[16]; int n = tokenize(prompt, ids);
    printf("tokens:"); for (int i = 0; i < n; i++) printf(" %d", ids[i]); printf("\n");
    if (n == 0) { printf("empty prompt\n"); return -1; }
    static float ctext[2 * 256];
    text_encode(ids, n, ctext);
    memcpy(ctext + M.D, M.null, M.D * 4);
    int64_t t1 = esp_timer_get_time();
    const int NZ = M.ZC * M.ZH * M.ZH;
    static float z[1024], v[2048];
    rng_x = (uint32_t)(seed * 2654435761u + 1u); if (!rng_x) rng_x = 1;
    gauss(z, NZ);
    for (int i = 0; i < steps && (g_stop_after < 0 || i < g_stop_after); i++) {
        float a = 1.f - (float)i / steps, b = 1.f - (float)(i + 1) / steps;
        float ta = shift * a / (1 + (shift - 1) * a), tb = shift * b / (1 + (shift - 1) * b);
        dit_forward(z, ta, ctext, v);
        for (int k = 0; k < NZ; k++) {
            float vc = v[k], vu = v[NZ + k];
            z[k] += (tb - ta) * (vu + cfg * (vc - vu));
        }
        printf("step %d/%d t=%.3f  %lld ms\n", i + 1, steps, ta, (esp_timer_get_time() - t1) / 1000);
    }
    int64_t t2 = esp_timer_get_time();
    decode(z, rgb);
    int64_t t3 = esp_timer_get_time();
    prof.text = t1 - t0; prof.dit = t2 - t1; prof.dec = t3 - t2; prof.total = t3 - t0;
    return 0;
}
