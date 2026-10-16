// Протокол по USB-Serial-JTAG:
//   -> "GEN seed steps cfg shift dual fast|промпт\n"
//   <- лог, "PROF ...", "IMG <base64 RGB 64x64>"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "driver/usb_serial_jtag.h"
#include "driver/usb_serial_jtag_vfs.h"
#include "esp_heap_caps.h"
#include "engine.h"

static const char B64[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
static void print_b64(const uint8_t *d, int n) {
    char line[4 * 64 + 1];
    int li = 0;
    for (int i = 0; i < n; i += 3) {
        uint32_t v = d[i] << 16 | (i + 1 < n ? d[i + 1] << 8 : 0) | (i + 2 < n ? d[i + 2] : 0);
        line[li++] = B64[(v >> 18) & 63]; line[li++] = B64[(v >> 12) & 63];
        line[li++] = i + 1 < n ? B64[(v >> 6) & 63] : '=';
        line[li++] = i + 2 < n ? B64[v & 63] : '=';
        if (li == 4 * 64) { line[li] = 0; fputs(line, stdout); li = 0; }
    }
    line[li] = 0; fputs(line, stdout);
}

void app_main(void) {
    usb_serial_jtag_driver_config_t cfg = USB_SERIAL_JTAG_DRIVER_CONFIG_DEFAULT();
    cfg.rx_buffer_size = 1024; cfg.tx_buffer_size = 4096;
    usb_serial_jtag_driver_install(&cfg);
    usb_serial_jtag_vfs_use_driver();
    setvbuf(stdin, NULL, _IONBF, 0);
    printf("free internal %u, psram %u\n", (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
           (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM));
    if (engine_load() || engine_init()) { printf("FATAL\n"); return; }
    static uint8_t rgb[64 * 64 * 3];
    static char buf[512];
    for (;;) {
        printf("READY\n"); fflush(stdout);
        int n = 0;
        while (n < (int)sizeof buf - 1) {
            int c = fgetc(stdin);
            if (c == EOF) { vTaskDelay(1); continue; }
            if (c == '\n') break;
            buf[n++] = c;
        }
        buf[n] = 0;
        if (strncmp(buf, "GEN ", 4)) continue;
        char *bar = strchr(buf, '|'); if (!bar) continue; *bar = 0;
        unsigned seed = 1; int steps = 8, dual = 1, fast = 1, pie = 1, stop = -1; float cfgs = 3.f, shift = 1.f;
        sscanf(buf + 4, "%u %d %f %f %d %d %d %d", &seed, &steps, &cfgs, &shift, &dual, &fast, &pie, &stop);
        g_stop_after = stop;
        engine_set(dual, fast, pie);
        printf("prompt: %s | seed %u steps %d cfg %.1f shift %.1f dual %d fast %d pie %d\n", bar + 1, seed, steps, cfgs, shift, dual, fast, pie);
        if (engine_generate(bar + 1, seed, steps, cfgs, shift, rgb)) continue;
        printf("PROF total=%lld text=%lld dit=%lld dec=%lld | gemm=%lld quant=%lld ln=%lld attn=%lld softmax=%lld gelu=%lld im2col=%lld dec_w=%lld dec_epi=%lld | macs=%lld\n",
               prof.total, prof.text, prof.dit, prof.dec, prof.gemm, prof.quant, prof.ln, prof.attn, prof.softmax,
               prof.gelu, prof.im2col, prof.dec_w, prof.dec_epi, prof.macs);
        printf("IMG "); print_b64(rgb, sizeof rgb); printf("\n"); fflush(stdout);
    }
}
