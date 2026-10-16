#!/usr/bin/env python3
# Клиент: отправляет промпт на ESP32-S3, получает картинку 64x64, сохраняет PNG и рисует её в терминале.
import argparse, base64, sys, time, re
import numpy as np
import serial
from PIL import Image

def find_port():
    # ESP32-S3 со встроенным USB-Serial/JTAG: VID 0x303A (Espressif)
    from serial.tools import list_ports
    for p in list_ports.comports():
        if p.vid == 0x303A:
            return p.device
    raise SystemExit("ESP32-S3 не найдена — укажите порт явно: --port COM5 / /dev/ttyACM0")

def connect(port=None):
    s = serial.Serial(port or find_port(), 115200, timeout=0.5)
    return s

def wait_ready(s, timeout=20, reset=False):
    if reset:
        s.dtr = False; s.rts = True; time.sleep(0.1); s.rts = False
    t = time.time(); buf = b""
    while time.time() - t < timeout:
        buf += s.read(4096)
        if b"READY" in buf: return buf.decode(errors="replace")
        if b"FATAL" in buf or b"Backtrace" in buf: break
    raise RuntimeError("no READY:\n" + buf.decode(errors="replace")[-3000:])

def generate(s, prompt, seed=1, steps=8, cfg=3.0, shift=1.0, dual=1, fast=1, log=True, timeout=300):
    s.reset_input_buffer()
    s.write(f"GEN {seed} {steps} {cfg} {shift} {dual} {fast}|{prompt}\n".encode())
    t = time.time(); buf = b""; info = {}
    while time.time() - t < timeout:
        chunk = s.read(8192)
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.decode(errors="replace").strip()
            if line.startswith("IMG "):
                rgb = np.frombuffer(base64.b64decode(line[4:]), np.uint8).reshape(64, 64, 3)
                return rgb, info
            if line.startswith("PROF"):
                info = {k: int(v) for k, v in re.findall(r"(\w+)=(\d+)", line)}
            if log and line: print("  esp>", line)
            if "Backtrace" in line or "Guru" in line: raise RuntimeError(line)
    raise RuntimeError("timeout\n" + buf.decode(errors="replace")[-2000:])

def show_term(rgb, scale=1):
    # полублоки ▀ с truecolor: 2 пикселя на символ
    im = rgb
    for y in range(0, im.shape[0], 2):
        row = []
        for x in range(im.shape[1]):
            a, b = im[y, x], im[y + 1, x]
            row.append(f"\x1b[38;2;{a[0]};{a[1]};{a[2]}m\x1b[48;2;{b[0]};{b[1]};{b[2]}m▀")
        print("".join(row) + "\x1b[0m")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("--port", default=None, help="по умолчанию — автопоиск ESP32-S3")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--cfg", type=float, default=3.0)
    ap.add_argument("--shift", type=float, default=1.0)
    ap.add_argument("--out", default="out.png")
    ap.add_argument("--reset", action="store_true")
    a = ap.parse_args()
    s = connect(a.port)
    if a.reset: wait_ready(s, reset=True)
    rgb, info = generate(s, a.prompt, a.seed, a.steps, a.cfg, a.shift)
    Image.fromarray(rgb).resize((256, 256), Image.NEAREST).save(a.out)
    show_term(rgb)
    print(f"готово за {info.get('total', 0) / 1e6:.1f} с, картинка 256x256 сохранена в {a.out}")
