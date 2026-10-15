#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAW Studio — аналіз .raw записів АЦП та генератор нових файлів у тому ж форматі.

Формат файлу:
    без заголовка; послідовність відліків uint16 little-endian,
    інтерлівінг ПО ВІДЛІКАХ: [ch0, ..., ch7, ch0, ...],
    типово 8 каналів; дефолтна fs = 12 ГГц (Найквіст 6 ГГц).

Частотне планування:
    6 блоків: 100-500, 500-1700, 1200-2500, 2400-3600, 3400-4600, 4400-6000 МГц
    каналізація: 64 МГц на канал → 105 повних + 5 крайових = 110 каналів.

Модуль АЧХ АЦП:
    амплітуда генерації залежно від частоти за профілем (дБ):
    плоский / спад −3 дБ/окт / смуговий 100М–6Г / провал (notch) / власна таблиця.
    Інтерполяція між точками: лінійна / кубічний сплайн / сплайн у log f;
    поза виміряним діапазоном: за краями / лінійна екстраполяція / нуль.
    Реалізація — FFT-фільтрація змінної складової; DC не модулюється.

Калібрування реальної АЧХ тракту:
    ступінчастий прогін тонів (кожен тон — у своєму сегменті, точно в біні FFT)
    → запис через реальний тракт → вимірювання амплітуди кожного тону →
    АЧХ тракту в дБ → збереження профілю в .achx → завантаження в Генератор.

Запуск:
    python3 raw_studio.py                         # графічний інтерфейс
    python3 raw_studio.py --headless FILE [fs] [nch]
    python3 raw_studio.py --selftest [FILE]       # самоперевірка ядра
    python3 raw_studio.py --preview OUT_PREFIX FILE [fs] [nch]
    python3 raw_studio.py --calgen OUT fs nch [tones fmin fmax amp seg]
    python3 raw_studio.py --calmeas RECORDED PLAN

Залежності: python3-tk, numpy, matplotlib.
"""
import argparse
import datetime
import json
import math
import os
import sys

import numpy as np

APP_TITLE = "RAW Studio — .raw АЦП: блоки 100М…6Г, канали 64 МГц, АЧХ + калібрування"
DEFAULT_FS = 12_000_000_000          # 12 ГГц
DEFAULT_NCH = 8
NFFT = 2 ** 20
SIG_MIN_HZ = 100e6
SIG_MAX_HZ = 6000e6
CHANNEL_BW = 64e6
FREQ_BLOCKS = [(100e6, 500e6), (500e6, 1700e6), (1200e6, 2500e6),
               (2400e6, 3600e6), (3400e6, 4600e6), (4400e6, 6000e6)]

SDR_PRESETS = {
    # fs/смуга/діапазон — з офіційних специфікацій виробників:
    # BladeRF 2.0 micro: nuand.com (47 МГц–6 ГГц, 61.44 МС/с, IBW 56 МГц);
    # HackRF One: hackrf.readthedocs.io (1 МГц–6 ГГц, 2–20 МС/с, 8 біт);
    # PlutoSDR: wiki.analog.com (325 МГц–3.8 ГГц, 61.44 МС/с, смуга 20 МГц);
    # LimeSDR USB: myriadrf.org (100 кГц–3.8 ГГц, 61.44 МС/с, смуга 61.44 МГц);
    # USRP B210: ettus.com (70 МГц–6 ГГц, 61.44 МС/с, IBW 56 МГц, AD9361);
    # HackRF Pro: greatscottgadgets.com (10 МГц–6 ГГц, 20 МС/с, 8 біт);
    # USRP X310 (UBX-160): kb.ettus.com (10 МГц–6 ГГц, 200 МС/с, IBW 160 МГц);
    # RTL-SDR Blog V4: rtl-sdr.com, datasheet V4 (500 кГц–1.766 ГГц,
    #                   2.56 МС/с стабільно / до 3.2 МС/с з просіданнями, 8 біт);
    # Aaronia SPECTRAN V6 X: downloads.aaronia.com (10 МГц–8 ГГц,
    #                   500 МС/с 16-біт, RTBW 120 МГц / до 245 МГц IQ);
    # Aaronia SPECTRAN V6 ECO: downloads.aaronia.com (9 кГц–6 ГГц,
    #                   500 МС/с 16-біт, RTBW 44 МГц / до 88 (2×44) МГц).
    "BladeRF 2.0 micro": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 47e6,
                          "f_hi": 6000e6, "bits": 12},
    "HackRF One": {"fs_hz": 20e6, "bw_hz": 20e6, "f_lo": 1e6,
                   "f_hi": 6000e6, "bits": 8},
    "ADALM-Pluto (61.44 МС/с)": {"fs_hz": 61.44e6, "bw_hz": 20e6, "f_lo": 325e6,
                                 "f_hi": 3800e6, "bits": 12},
    "LimeSDR USB": {"fs_hz": 61.44e6, "bw_hz": 61.44e6, "f_lo": 100e3, "f_hi": 3800e6, "bits": 12},
    "USRP B210": {"fs_hz": 61.44e6, "bw_hz": 56e6, "f_lo": 70e6, "f_hi": 6000e6, "bits": 12},
    "HackRF Pro": {"fs_hz": 20e6, "bw_hz": 20e6, "f_lo": 10e6, "f_hi": 6000e6, "bits": 8},
    "USRP X310 (UBX-160)": {"fs_hz": 200e6, "bw_hz": 160e6, "f_lo": 10e6, "f_hi": 6000e6, "bits": 12},
    "RTL-SDR Blog V4": {"fs_hz": 2.56e6, "bw_hz": 2.56e6, "f_lo": 500e3, "f_hi": 1766e6, "bits": 8},
    "Aaronia SPECTRAN V6 X": {"fs_hz": 500e6, "bw_hz": 120e6, "f_lo": 10e6, "f_hi": 8000e6, "bits": 16},
    "Aaronia SPECTRAN V6 ECO": {"fs_hz": 500e6, "bw_hz": 44e6, "f_lo": 9e3, "f_hi": 6000e6, "bits": 16},
    "Інший (власноруч)": None,
}

WAVE_KINDS = ["Синус", "Меандр", "Трикутник", "Імпульс", "Шум"]
KIND_KEY = {"Синус": "sine", "Меандр": "square", "Трикутник": "tri",
            "Імпульс": "pulse", "Шум": "noise"}

ACHX_PRESETS = ["Плоский", "Спад −3 дБ/окт", "Смуговий 100М–6Г",
                "Провал (notch)", "Користувацький"]
ACHX_MODES = {"Лінійна": "linear", "Кубічний сплайн": "spline", "Сплайн у log f": "logspline"}
ACHX_EXTRAPS = {"За краями (clamped)": "clamp", "Лінійно далі": "linear",
                "Нуль поза смугою": "zero"}


def human_freq(f):
    a = abs(f)
    if a >= 1e9:
        return f / 1e9, "ГГц"
    if a >= 1e6:
        return f / 1e6, "МГц"
    if a >= 1e3:
        return f / 1e3, "кГц"
    return f, "Гц"


def parse_freq_hz(s):
    """-> Гц. Голе число — МГц (основний режим); суфікси мгц/м, ггц/g, кгц/k,
    гц/hz — явні одиниці; число ≥ 1e6 (напр. 2.4e9, 12000000000,
    1.32e+08) трактується як Гц (старі записи), менше — як МГц."""
    t = str(s).strip().lower()
    for cy, la in (("мгц", "mhz"), ("ггц", "ghz"), ("кгц", "khz"), ("гц", "hz"),
                   ("м", "m"), ("г", "g"), ("к", "k")):
        if t.endswith(cy):
            t = t[:-len(cy)] + la
            break
    for suf, mult in (("mhz", 1e6), ("ghz", 1e9), ("khz", 1e3), ("hz", 1.0),
                      ("m", 1e6), ("g", 1e9), ("k", 1e3)):
        if t.endswith(suf):
            return float(t[:-len(suf)]) * mult
    v = float(t)
    if abs(v) >= 1e6:
        return v
    return v * 1e6


def freq_unit_for(fs):
    half = fs / 2.0
    if half >= 1e9:
        return 1e9, "ГГц"
    if half >= 1e6:
        return 1e6, "МГц"
    if half >= 1e3:
        return 1e3, "кГц"
    return 1.0, "Гц"


def build_blocks():
    """Решітка блоків і каналів по 64 МГц; нумерація каналів наскрізна."""
    blocks = []
    g = 0
    for bi, (f0, f1) in enumerate(FREQ_BLOCKS, 1):
        chs = []
        c = 0
        f = f0
        while f < f1 - 1.0:
            w = min(CHANNEL_BW, f1 - f)
            g += 1
            c += 1
            chs.append({"global": g, "in_block": c, "f0": f, "f1": f + w,
                        "fc": f + w / 2.0, "w": w, "partial": w < CHANNEL_BW - 1.0})
            f += w
        blocks.append({"idx": bi, "f0": f0, "f1": f1, "channels": chs})
    return blocks


# ----------------------------------------------------------------------------
# МОДУЛЬ АЧХ АЦП — амплітуда генерації залежно від частоти
# ----------------------------------------------------------------------------
def build_achx_points(preset, notch_hz=1e9):
    """Профіль АЧХ: список точок (freq_Hz, gain_dB)."""
    f0, f1 = SIG_MIN_HZ, SIG_MAX_HZ
    if preset == "Плоский":
        return [(0.0, 0.0), (1e12, 0.0)]
    if preset == "Спад −3 дБ/окт":
        pts = [(0.0, 0.0)]
        for i in range(61):
            f = f0 * (f1 / f0) ** (i / 60.0)
            pts.append((f, -3.0 * math.log2(f / f0)))
        pts.append((1e12, -3.0 * math.log2(1e12 / f0)))
        return pts
    if preset == "Смуговий 100М–6Г":
        return [(0.0, -60.0), (f0, -3.0), (f0 * 1.02, 0.0),
                (f1 * 0.98, 0.0), (f1, -3.0), (1.2e12, -60.0)]
    if preset == "Провал (notch)":
        return [(0.0, 0.0), (notch_hz * 0.9, 0.0), (notch_hz, -40.0),
                (notch_hz * 1.1, 0.0), (1e12, 0.0)]
    raise ValueError("невідомий профіль: %s" % preset)


def parse_achx_table(text):
    """Рядки «частота gain_дБ»: '100e6 0', '1e9, -3', '6e9; -12' …"""
    pts = []
    for line in text.strip().splitlines():
        line = line.replace(",", " ").replace(";", " ").strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError("рядок АЧХ '%s' — потрібно: частота дБ" % line)
        pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        raise ValueError("АЧХ: мінімум 2 точки")
    pts.sort()
    return pts


# --- інтерполяція профілю: лінійна / кубічний сплайн / сплайн у log f ---------
def _spline_second_derivs(x, y):
    """Другі похідні природного кубічного сплайна (M[0]=M[-1]=0)."""
    n = len(x)
    h = np.diff(x)
    A = np.zeros((n, n)); rhs = np.zeros(n)
    A[0, 0] = 1.0; A[-1, -1] = 1.0
    for i in range(1, n - 1):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        rhs[i] = 6.0 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])
    return np.linalg.solve(A, rhs)


def _spline_eval(x, y, M, xq, extrap="clamp"):
    """Значення сплайна в точках xq; екстраполяція: clamp | linear."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xq = np.asarray(xq, dtype=np.float64)
    out = np.empty(xq.shape, dtype=np.float64)
    if extrap == "clamp":
        xqc = np.clip(xq, x[0], x[-1])
    else:
        xqc = xq.copy()
    inside = (xqc >= x[0]) & (xqc <= x[-1])
    xi = xqc[inside]
    i = np.clip(np.searchsorted(x, xi, side="right") - 1, 0, len(x) - 2)
    h = np.diff(x)[i]
    t = (xi - x[i]) / h
    out[inside] = (y[i] * (1 - t) + y[i + 1] * t
                   + h * h / 6.0 * (M[i] * ((1 - t) ** 3 - (1 - t))
                                    + M[i + 1] * (t ** 3 - t)))
    if extrap == "linear":
        m_hi = (y[-1] - y[-2]) / (x[-1] - x[-2])
        m_lo = (y[1] - y[0]) / (x[1] - x[0])
        out[xq > x[-1]] = y[-1] + m_hi * (xq[xq > x[-1]] - x[-1])
        out[xq < x[0]] = y[0] + m_lo * (xq[xq < x[0]] - x[0])
    return out


def achx_response(freqs, points, mode="linear", extrap="clamp"):
    """|H(f)| у лінійних одиницях з профілю (дБ).
    mode:  linear — np.interp; spline — кубічний сплайн у Гц;
           logspline — кубічний сплайн у log10(f).
    extrap: clamp — значення країв; linear — продовження крайового нахилу;
            zero — 0 (−120 дБ) поза виміряним діапазоном.
    Оцінка виконується на БУДЬ-ЯКІЙ частоті (в т.ч. між точками профілю)."""
    fr = np.array([p[0] for p in points], dtype=np.float64)
    db = np.array([p[1] for p in points], dtype=np.float64)
    fq = np.atleast_1d(np.asarray(freqs, dtype=np.float64))

    if mode == "logspline":
        xs = np.log10(fr)
        xq = np.log10(np.clip(fq, 1e-6, None))
    else:
        xs, xq = fr, fq.copy()

    use_spline = mode in ("spline", "logspline") and len(fr) >= 3
    if use_spline:
        M = _spline_second_derivs(xs, db)
        hdb = _spline_eval(xs, db, M, xq,
                           extrap=("clamp" if extrap == "zero" else extrap))
    else:
        hdb = np.interp(xq, xs, db)                      # лінійна, clamp
        if extrap == "linear" and len(fr) >= 2:
            m_lo = (db[1] - db[0]) / (fr[1] - fr[0])
            m_hi = (db[-1] - db[-2]) / (fr[-1] - fr[-2])
            hdb = np.where(fq < fr[0], db[0] + m_lo * (fq - fr[0]), hdb)
            hdb = np.where(fq > fr[-1], db[-1] + m_hi * (fq - fr[-1]), hdb)
    if extrap == "zero":
        outside = (fq < fr[0]) | (fq > fr[-1])
        hdb = np.where(outside, -120.0, hdb)
    out = 10.0 ** (hdb / 20.0)
    if extrap == "zero":
        out = np.where(outside, 0.0, out)   # справжній нуль підсилення поза смугою
    return out


def apply_achx(w, fs, points, mode="linear", extrap="clamp"):
    """Застосовує АЧХ до змінної складової сигнала (FFT-фільтрація)."""
    n = w.size
    W = np.fft.rfft(w)
    f = np.fft.rfftfreq(n, d=1.0 / fs)
    H = achx_response(f, points, mode=mode, extrap=extrap)
    return np.fft.irfft(W * H, n)


# ----------------------------------------------------------------------------
# КАЛІБРУВАННЯ РЕАЛЬНОЇ АЧХ ТРАКТУ (ступінчастий прогін тонів)
# ----------------------------------------------------------------------------
def make_cal_plan(fs, f_min, f_max, n_tones, seg_n):
    """Частоти тонів, кожна округлена до цілих бінів FFT. → (freqs, bin_hz)."""
    bin_hz = fs / float(seg_n)
    f0s = np.linspace(f_min, f_max, n_tones)
    freqs = np.round(f0s / bin_hz) * bin_hz
    freqs = np.unique(freqs)
    freqs = np.clip(freqs, bin_hz, fs / 2.0 - bin_hz)
    return freqs.tolist(), bin_hz


def generate_cal_file(fs, nch, freqs, amp, seg_n, tract_points=None):
    """Ступінчастий прогін: сегмент seg_n відликів на тон (у ch0, DC 32768).
    tract_points — емуляція тракту (масштабує ТІЛЬКИ тон, не DC)."""
    n = seg_n * len(freqs)
    cols = [np.zeros(n) for _ in range(nch)]
    tloc = np.arange(seg_n) / fs
    for i, fq in enumerate(freqs):
        scale = 1.0 if tract_points is None else float(achx_response(np.array([fq]), tract_points)[0])
        seg = 32768.0 + amp * scale * np.sin(2.0 * np.pi * fq * tloc)
        cols[0][i * seg_n:(i + 1) * seg_n] = seg
    block = np.stack(cols, axis=1)
    block = np.clip(np.rint(block), 0, 65535).astype("<u2")
    return block.reshape(-1)


def save_cal_plan(path, fs, nch, freqs, amp, seg_n, meas_ch=0):
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"fs": fs, "nch": nch, "freqs": freqs, "amp": amp,
                   "seg_n": seg_n, "meas_ch": meas_ch}, fh)


def load_cal_plan(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def measure_achx(path, plan, meas_ch=None):
    """Вимірювання реальної АЧХ: амплітуда кожного тону в своєму сегменті.
    → [(freq_Hz, amp_codes, gain_dB_відносно_еталону), ...]"""
    fs = plan["fs"]
    freqs = plan["freqs"]
    amp_ref = plan["amp"]
    seg_n = plan["seg_n"]
    k = plan["meas_ch"] if meas_ch is None else meas_ch
    chans, n = load_raw(path, plan["nch"])
    need = seg_n * len(freqs)
    if n < need:
        raise ValueError("запис коротший за план: %d < %d відліків" % (n, need))
    s = chans[k]
    out = []
    for i, fq in enumerate(freqs):
        seg = s[i * seg_n:(i + 1) * seg_n]
        f, a = amplitude_spectrum(seg, fs, nfft=seg_n)
        idx = int(round(fq / (fs / seg_n)))
        amp = float(a[idx])
        out.append((float(fq), amp, 20.0 * math.log10(max(amp, 1e-9) / amp_ref)))
    return out


def save_achx_profile(path, points, meta=None, mode=None, extrap=None):
    """Профіль АЧХ у текстовий .achx: рядки «частота_Гц дБ», # — коментарі;
    режим інтерполяції/екстраполяції зберігається в шапці."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# RAW Studio — профіль АЧХ (частота_Гц дБ)\n")
        if mode:
            fh.write("# interpolation: %s\n" % mode)
        if extrap:
            fh.write("# extrapolation: %s\n" % extrap)
        for k, v in (meta or {}).items():
            fh.write("# %s: %s\n" % (k, v))
        for fq, gdb in points:
            fh.write("%.6g %.4f\n" % (fq, gdb))


def load_achx_profile(path):
    """Читає .achx → (points, mode, extrap); режими — з шапки або дефолтні."""
    mode, extrap = None, None
    pts = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            t = line.strip()
            if t.startswith("# interpolation:"):
                mode = t.split(":", 1)[1].strip()
            elif t.startswith("# extrapolation:"):
                extrap = t.split(":", 1)[1].strip()
            elif t and not t.startswith("#"):
                parts = t.replace(",", " ").replace(";", " ").split()
                if len(parts) >= 2:
                    pts.append((float(parts[0]), float(parts[1])))
    if len(pts) < 2:
        raise ValueError("АЧХ: мінімум 2 точки в файлі")
    pts.sort()
    mode = {"log": "logspline"}.get(mode, mode)   # зворотна сумісність старих .achx
    return pts, (mode or "linear"), (extrap or "clamp")


# ----------------------------------------------------------------------------
# ЯДРО (без GUI) — завантаження, статистика, спектр, генерація
# ----------------------------------------------------------------------------
def load_raw(path, nch=DEFAULT_NCH):
    data = np.fromfile(path, dtype="<u2")
    if data.size == 0:
        raise ValueError("порожній файл: %s" % path)
    n = data.size // nch
    if n * nch != data.size:
        print("  ! розмір не кратний %d каналам — останні %d байт(ів) відкинуто"
              % (nch, (data.size % nch) * 2))
    data = data[:n * nch].reshape(n, nch)
    return {k: data[:, k].astype(np.float64) for k in range(nch)}, n


def channel_stats(s, fs):
    hi = float((s >= 65000).mean() * 100.0)
    lo = float((s <= 500).mean() * 100.0)
    return {"dc": float(s.mean()), "rms": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "clip_hi": hi, "clip_lo": lo}


def amplitude_spectrum(s, fs, nfft=NFFT):
    """Пікова амплітуда в кодах АЦП: A = 2|FFT|/(N·CG), вікно Ганна, детренд."""
    s = s[:nfft]
    w = np.hanning(s.size)
    cg = w.mean()
    x = s - s.mean()
    a = np.abs(np.fft.rfft(x * w)) * 2.0 / (s.size * cg)
    f = np.fft.rfftfreq(s.size, d=1.0 / fs)
    return f, a


def find_peaks(f, a, k=5, exclude_bins=5, min_sep_bins=8):
    order = np.argsort(a)[::-1]
    taken, out = [], []
    for i in order:
        if i < exclude_bins:
            continue
        if any(abs(int(i) - t) < min_sep_bins for t in taken):
            continue
        taken.append(int(i))
        out.append((float(f[i]), float(a[i])))
        if len(out) >= k:
            break
    return out


def channel_occupancy(chans, fs, blocks, factor=6.0):
    power = None
    for k in sorted(chans):
        f, a = amplitude_spectrum(chans[k], fs)
        p = a.astype(np.float64) ** 2
        power = p if power is None else power + p
    raw = {}
    for b in blocks:
        for ch in b["channels"]:
            m = (f >= ch["f0"]) & (f < ch["f1"])
            raw[ch["global"]] = float(power[m].mean()) if m.any() else 0.0
    med = float(np.median(list(raw.values())))
    out = {}
    for g, pw in raw.items():
        out[g] = {"power": pw, "occupied": pw > factor * med and pw > 0}
    return out


def make_wave(kind_key, n, fs, freq, phase_deg, amp, width_us, rng):
    t = np.arange(n) / fs
    if kind_key == "noise":
        return rng.normal(0.0, amp, n)
    if freq <= 0:
        return np.zeros(n)
    period = 1.0 / freq
    phase = (t - phase_deg / 360.0 / freq) % period
    hi_t = min(max(width_us * 1e-6, 0.0), period)
    if kind_key == "sine":
        return amp * np.sin(2.0 * np.pi * freq * t + math.radians(phase_deg))
    if kind_key == "square":
        return np.where(phase < hi_t, amp, -amp)
    if kind_key == "pulse":
        return np.where(phase < hi_t, amp, 0.0)
    if kind_key == "tri":
        x = 2.0 * np.abs(2.0 * phase / period - 1.0) - 1.0
        return amp * x
    raise ValueError(kind_key)


def make_wave_ext(kind_key, n, fs, freq, phase_deg, amp, width_us, rng,
                  per_us=0.0, pw_us=0.0, rise_us=0.0, fall_us=0.0,
                  am_m=0.0, am_f=0.0, chirp_dev=0.0):
    """Розширені форми: радіоімпульси (період слідування, фронти), АМ, ЛЧМ.
    Амплітуда/фаза — базові; тут додаткові параметри сигналів."""
    t = np.arange(n) / fs
    width = pw_us if pw_us > 0 else width_us
    per = per_us * 1e-6 if per_us > 0 else 0.0
    use_chirp = chirp_dev > 0 and freq > 0
    if kind_key in ("tri", "noise"):
        w = make_wave(kind_key, n, fs, freq, phase_deg, amp, width_us, rng)
    elif kind_key == "pulse":
        # радіоімпульс: несуча (Частота) × огинаюча з фронтами зростання/спаду
        hi = width * 1e-6
        if freq <= 0:
            w = np.zeros(n)
        else:
            pp = per if per > 0 else max(1.0 / freq, hi * 1.25)
            tloc = np.mod(t, pp)
            ph = 2 * np.pi * freq * t + math.radians(phase_deg)
            if use_chirp:
                kk = chirp_dev / max(hi, 1e-12)
                ph = 2 * np.pi * (freq * t + 0.5 * kk * np.minimum(tloc, hi) ** 2) \
                     + math.radians(phase_deg)
            env = np.zeros(n)
            on = tloc < hi
            env[on] = 1.0
            rise = min(max(rise_us * 1e-6, 0.0), hi / 2)
            fall = min(max(fall_us * 1e-6, 0.0), hi - rise)
            if rise > 0:
                r = tloc < rise
                env[r] = tloc[r] / rise
            if fall > 0:
                fl = (tloc >= hi - fall) & on
                env[fl] = np.maximum((hi - tloc[fl]) / fall, 0.0)
            w = amp * np.sin(ph) * env
    elif kind_key == "square":
        # меандр: трапеція з фронтами зростання/спаду
        hi = width * 1e-6
        pp = per if per > 0 else 2 * hi
        tloc = np.mod(t, pp)
        w = np.where(tloc < hi, float(amp), -float(amp))
        rise = min(max(rise_us * 1e-6, 0.0), hi / 2)
        fall = min(max(fall_us * 1e-6, 0.0), hi / 2)
        if rise > 0:
            r = tloc < rise
            w[r] = -amp + 2 * amp * tloc[r] / rise
        if fall > 0:
            fl = (tloc >= hi - fall) & (tloc < hi)
            w[fl] = amp - 2 * amp * (tloc[fl] - (hi - fall)) / fall
    elif kind_key == "sine":
        if use_chirp:
            kk = chirp_dev / (n / fs)
            ph = 2 * np.pi * (freq * t + 0.5 * kk * t ** 2) + math.radians(phase_deg)
            w = amp * np.sin(ph)
        else:
            w = amp * np.sin(2 * np.pi * freq * t + math.radians(phase_deg))
    else:
        raise ValueError(kind_key)
    if am_m > 0 and am_f > 0:
        w = w * (1.0 + am_m * np.sin(2 * np.pi * am_f * t))
    return w


def generate_raw(ch_params, n_samples, fs, nch, seed=42, achx=None, extra=None):
    """Створює файл у форматі вхідного.
    achx: None або {"points": [(Гц, дБ)...], "mode": linear|spline|log,
    "extrap": clamp|linear|zero} — застосовується до змінної складової
    КОЖНОГО увімкненого канала на БУДЬ-ЯКІЙ частоті (інтерполяція)."""
    rng = np.random.default_rng(seed)
    cols = []
    for k in range(nch):
        p = ch_params[k]
        n = n_samples
        if not p.get("enabled"):
            cols.append(np.full(n, p.get("dc", 32768.0)))
            continue
        ex = extra or {}
        if ex and any(float(ex.get(k) or 0) != 0 for k in
                      ("per_us", "pw_us", "rise_us", "fall_us", "am_m", "am_f", "chirp_dev")):
            w = make_wave_ext(KIND_KEY[p["kind"]], n, fs, p["freq"], p["phase"],
                              p["amp"], p["width_us"], rng,
                              per_us=ex.get("per_us", 0), pw_us=ex.get("pw_us", 0),
                              rise_us=ex.get("rise_us", 0), fall_us=ex.get("fall_us", 0),
                              am_m=ex.get("am_m", 0), am_f=ex.get("am_f", 0),
                              chirp_dev=ex.get("chirp_dev", 0))
        else:
            w = make_wave(KIND_KEY[p["kind"]], n, fs, p["freq"], p["phase"],
                          p["amp"], p["width_us"], rng)
        if achx and achx.get("points"):
            w = apply_achx(w, fs, achx["points"],
                           mode=achx.get("mode", "linear"),
                           extrap=achx.get("extrap", "clamp"))
        s = p.get("dc", 32768.0) + p.get("gain", 1.0) * w
        cols.append(s)
    block = np.stack(cols, axis=1)
    block = np.clip(np.rint(block), 0, 65535).astype("<u2")
    return block.reshape(-1)


def save_raw(path, flat_u16):
    flat_u16.tofile(path)


RTSA_IQ_DTYPES = {"float32": "<f4", "float64": "<f8", "int8": "i1",
                  "int16": "<i2", "int32": "<i4"}
RTSA_IQ_NORM = {"float32": 1.0, "float64": 1.0, "int8": 128.0,
                "int16": 32768.0, "int32": 2147483648.0}


SIG_PRESETS = {
    # Типові параметри стандартів — з відкритих специфікацій:
    # GSM: ETSI GSM 05.01 / sharetechnote (кадр 4.615 мс, слот 577 мкс);
    # GPS L1: gps.gov SPS Signal Spec (1575.42 МГц, BPSK 1.023 Мсимв/с,
    #         2.046 МГц між нулями);
    # LoRa: 125/250/500 кГц CSS; WCDMA: 3.84 Мсимв/с, 5 МГц;
    # LTE: OFDMA до 20 МГц; NR FR1: до 100 МГц; NB-IoT: 180 кГц;
    # DVB-T/T2: 8 МГц; Bluetooth FHSS: 79 каналів × 1 МГц; CVBS: 64 мкс/4.7 мкс.
    "LoRa": {"kind": "Синус", "freq": 868e6, "width_us": None,
             "extra": {"chirp_dev": 0.125e6}, "fs_hint": 2000e6,
             "label": "868 МГц, ЛЧМ 125 кГц",
             "note": "LoRa (CSS): чірип-спектр BW 125 кГц (SF7–SF12); модель — несуча з ЛЧМ 125 кГц"},
    "LoRa VHF (150–300)": {"kind": "Синус", "freq": 169e6, "width_us": None,
                           "extra": {"chirp_dev": 0.125e6}, "fs_hint": 500e6,
                           "label": "169 МГц (150–300), ЛЧМ 125 кГц",
                           "note": "LoRa VHF: діапазон 150–300 МГц (реальний SRD-діапазон LoRa — 169 МГц); несучу можна змінити на будь-яку з 150–300 у полі «Частота»; модель — несуча з ЛЧМ 125 кГц"},
    "LoRa 433 МГц": {"kind": "Синус", "freq": 433.92e6, "width_us": None,
                     "extra": {"chirp_dev": 0.125e6}, "fs_hint": 1000e6,
                     "label": "433.92 МГц, ЛЧМ 125 кГц",
                     "note": "LoRa 433: ISM 433.05–434.79 МГц (LoRaWAN EU433: 433.175–434.665 МГц); модель — несуча з ЛЧМ 125 кГц"},
    "LoRaWAN": {"kind": "Імпульс", "freq": 868.3e6, "width_us": None,
                "extra": {"per_us": 10000, "pw_us": 1000, "rise_us": 5, "fall_us": 5,
                          "chirp_dev": 0.125e6}, "fs_hint": 2000e6,
                "label": "пакети 1 мс / 10 мс, ЛЧМ 125 кГц",
                "note": "LoRaWAN: LoRa-фірика пакетами; модель — радіоімпульси 1 мс з періодом 10 мс, ЛЧМ 125 кГц"},
    "2G (GSM)": {"kind": "Імпульс", "freq": 900e6, "width_us": None,
                 "extra": {"per_us": 4615, "pw_us": 577, "rise_us": 3, "fall_us": 3},
                 "fs_hint": 2000e6,
                 "label": "слот 577 мкс / кадр 4.615 мс",
                 "note": "2G (GSM): TDMA GMSK, кадр 4.615 мс = 8 слотів по 577 мкс; модель — радіоімпульси з фронтами 3 мкс"},
    "3G (WCDMA)": {"kind": "Шум", "freq": 2140e6, "width_us": None,
                   "extra": {}, "fs_hint": 5000e6,
                   "label": "канал 5 МГц, QPSK",
                   "note": "3G (WCDMA): QPSK 3.84 Мсимв/с, канал 5 МГц; модель — широкосмуговий шум (2140 МГц)"},
    "4G (LTE)": {"kind": "Шум", "freq": 2600e6, "width_us": None,
                 "extra": {}, "fs_hint": 6000e6,
                 "label": "OFDMA до 20 МГц",
                 "note": "4G (LTE): OFDMA, канал до 20 МГц; модель — широкосмуговий шум (2600 МГц)"},
    "5G (NR)": {"kind": "Шум", "freq": 3500e6, "width_us": None,
                "extra": {}, "fs_hint": 8000e6,
                "label": "n78 (3.3–3.8 ГГц), до 100 МГц",
                "note": "5G (NR): OFDM FR1, діапазон n78, канал до 100 МГц; модель — широкосмуговий шум (3500 МГц)"},
    "WiFi (OFDM)": {"kind": "Шум", "freq": 2437e6, "width_us": None,
                    "extra": {"am_m": 0.9, "am_f": 1000}, "fs_hint": 6000e6,
                    "label": "20/40/80 МГц, пакети",
                    "note": "WiFi: OFDM 20/40/80 МГц, пакетна передача; модель — шум з АМ-огинаючою пакетів (1 кГц)"},
    "GNSS (L-смуга)": {"kind": "Шум", "freq": 1580e6, "width_us": None,
                       "extra": {}, "fs_hint": 4000e6,
                       "label": "1559–1610 МГц",
                       "note": "GNSS: спредспектр BPSK у смузі 1559–1610 МГц; модель — широкосмуговий шум"},
    "GPS L1": {"kind": "Шум", "freq": 1575.42e6, "width_us": None,
               "extra": {}, "fs_hint": 4000e6,
               "label": "L1 C/A, 2.046 МГц",
               "note": "GPS L1 C/A: BPSK 1.023 Мсимв/с (2.046 МГц між нулями), 1575.42 МГц; модель — шум"},
    "UNB / Narrowband": {"kind": "Синус", "freq": 868.13e6, "width_us": None,
                         "extra": {}, "fs_hint": 2000e6,
                         "label": "100/600 Гц (Sigfox)",
                         "note": "UNB (Sigfox): DBPSK, смуга 100/600 Гц; модель — вузькосмугова несуча (868.13 МГц)"},
    "FSK / GFSK": {"kind": "Синус", "freq": 2440e6, "width_us": None,
                   "extra": {"chirp_dev": 0.5e6}, "fs_hint": 6000e6,
                   "label": "2.4 ГГц, dev 0.5 МГц",
                   "note": "FSK/GFSK (Bluetooth): частотна манипуляція dev ±160…500 кГц; модель — несуча з частотною модуляцією (ЛЧМ 0.5 МГц)"},
    "FHSS": {"kind": "Шум", "freq": 2441.75e6, "width_us": None,
             "extra": {}, "fs_hint": 6000e6,
             "label": "79×1 МГц, 1600 стрибків/с",
             "note": "FHSS (Bluetooth): 79 каналів × 1 МГц, 1600 стрибків/с; модель — шум у смузі 2.4 ГГц (усереднено)"},
    "QAM": {"kind": "Шум", "freq": 500e6, "width_us": None,
            "extra": {}, "fs_hint": 2000e6,
            "label": "64/256-QAM",
            "note": "QAM (64/256): амплітудно-фазова модуляція, високий PAPR; модель — шумове заповнення канала (500 МГц)"},
    "CVBS": {"kind": "Меандр", "freq": 0, "width_us": 4.7,
             "extra": {"per_us": 64, "rise_us": 0.2, "fall_us": 0.2}, "fs_hint": 100e6,
             "label": "рядок 64 мкс, sync 4.7 мкс",
             "note": "CVBS: аналогове відео, рядок 64 мкс (15.625 кГц), синхроімпульс 4.7 мкс; модель — трапеція в базовій смузі (несуча 0)"},
    "COFDM (DVB)": {"kind": "Шум", "freq": 474e6, "width_us": None,
                    "extra": {}, "fs_hint": 2000e6,
                    "label": "DVB-T/T2, 8 МГц",
                    "note": "COFDM (DVB-T/T2): канал 8 МГц; модель — широкосмуговий шум (UHF 474 МГц)"},
    "NB-IoT": {"kind": "Синус", "freq": 869.5e6, "width_us": None,
               "extra": {}, "fs_hint": 2000e6,
               "label": "канал 180 кГц",
               "note": "NB-IoT: SC-FDMA, канал 180 кГц, напівдуплекс; модель — вузькосмугова несуча (869.5 МГц)"},
}


def parse_rtsa_xml(xml_path):
    """Sidecar .iq.xml запису Raw IQ з RTSA-Suite (RTSA_RAW_IQ_File):
    Clock (Гц), CenterFrequency (Гц), DataType (float32/int16/...)."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.parse(xml_path).getroot()
    except Exception:
        return None
    if "RTSA_RAW_IQ_File" not in (root.tag or ""):
        return None

    def _num(tag):
        el = root.find(tag)
        if el is None or not el.text:
            return None
        try:
            return float(el.text.strip())
        except ValueError:
            return None

    out = {}
    clock, cf = _num("Clock"), _num("CenterFrequency")
    if clock:
        out["fs_iq"] = clock
    if cf is not None:
        out["fc"] = cf
    dt = root.findtext("DataType")
    if dt and dt.strip():
        out["dtype"] = dt.strip()
    return out or None


def read_iq(path, dtype="float32", fs_iq=0.0, fc=0.0, max_samples=1 << 22):
    """Читає I/Q запис: .iq (raw інтерлівінг Re,Im), .wav (стерео I=лівий,
    Q=правий) або будь-який бінарний raw. Повертає (комплексний масив, meta)."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        import wave
        w = wave.open(path, "rb")
        nch_w, sw, fr, nf = (w.getnchannels(), w.getsampwidth(),
                             w.getframerate(), w.getnframes())
        raw = w.readframes(nf)
        w.close()
        if nch_w != 2:
            raise ValueError("WAV має бути стерео (I — лівий, Q — правий канал)")
        if sw == 2:
            s = np.frombuffer(raw, "<i2").astype(np.float64) / 32768.0
            dts = "int16(wav)"
        elif sw == 4:
            s = np.frombuffer(raw, "<f4").astype(np.float64)
            dts = "float32(wav)"
        else:
            raise ValueError("WAV: підтримуються 16-біт int і 32-біт float")
        if s.size % 2:
            s = s[:-1]
        x = s[0::2] + 1j * s[1::2]
        return x[:max_samples], {"fs_iq": fs_iq or float(fr), "fc": fc,
                                 "dtype": dts, "n_total": nf}
    dt = RTSA_IQ_DTYPES.get(dtype)
    if dt is None:
        raise ValueError("Невірний тип даних: %s" % dtype)
    a = np.fromfile(path, dt)
    if a.size < 2:
        raise ValueError("Файл занадто малий або порожній")
    if a.size % 2:
        a = a[:-1]
    a = a[:2 * max_samples]
    x = a[0::2].astype(np.float64) + 1j * a[1::2].astype(np.float64)
    x = x / RTSA_IQ_NORM[dtype]
    return x, {"fs_iq": fs_iq, "fc": fc, "dtype": dtype,
               "n_total": a.size // 2}


def plan_channels():
    """Канали частотного плану 100М…6Г по 64 МГц: [(f0, f1, global), ...]."""
    out, g = [], 0
    for lo, hi in FREQ_BLOCKS:
        f0 = lo
        while f0 + CHANNEL_BW <= hi + 1e-6:
            out.append((f0, f0 + CHANNEL_BW, g))
            g += 1
            f0 += CHANNEL_BW
    return out


def iq_to_raw(x, fc, fs_iq, fs_out=DEFAULT_FS, n_per_ch=16384, nch=8, seed=42):
    """Конвертує I/Q запис у .raw: середній спектр (Уелч) розкладається по
    каналах плану 64 МГц; кожен канал плану із сигналом синтезується як
    часовий ряд (випадкові фази, профіль = запис) на частоті fs_out.
    Повна шкала запису (комплексна амплітуда 1.0) = несуча 32768 кодів."""
    if fs_iq <= 0 or fc <= 0:
        raise ValueError("Вкажіть fs I/Q і центральну частоту (з .iq.xml або вручну)")
    n = x.size
    nfft_iq = min(8192, n)
    if nfft_iq < 16:
        raise ValueError("Запис I/Q занадто короткий")
    w = np.hanning(nfft_iq)
    hop = nfft_iq // 2
    nseg = int(min(200, max(1, (n - nfft_iq) // hop + 1)))
    P = np.zeros(nfft_iq)
    for s in range(nseg):
        seg = x[s * hop: s * hop + nfft_iq] * w
        P += np.abs(np.fft.fftshift(np.fft.fft(seg))) ** 2
    if not np.isfinite(P.sum()) or P.sum() <= 0:
        raise ValueError("Запис не містить сигналу")
    p_share = P / P.sum()
    fbin = fc + np.fft.fftshift(np.fft.fftfreq(nfft_iq, 1.0 / fs_iq))
    p_tot = float((np.abs(x) ** 2).mean())
    assign = []
    for gi, (f0, f1, g) in enumerate(plan_channels()):
        i0, i1 = np.searchsorted(fbin, f0), np.searchsorted(fbin, f1)
        sh = float(p_share[i0:i1].sum())
        if sh > 0:
            assign.append((g, f0, f1, sh))
    n_syn = max(1024, int(n_per_ch) // 2 * 2)
    half = n_syn // 2 + 1
    delta = fs_out / n_syn
    rng = np.random.default_rng(seed)
    cols = [None] * nch
    for idx, (g, f0, f1, sh) in enumerate(assign[:nch]):
        i0, i1 = np.searchsorted(fbin, f0), np.searchsorted(fbin, f1)
        f_rec, p_rec = fbin[i0:i1], p_share[i0:i1]
        j0 = max(1, int(np.ceil(f0 / delta)))
        j1 = min(half - 2, int(np.floor(f1 / delta)))
        if j1 < j0 or f_rec.size == 0:
            continue
        centers = np.arange(j0, j1 + 1) * delta
        w_lo = np.searchsorted(f_rec, centers - delta / 2.0)
        w_hi = np.searchsorted(f_rec, centers + delta / 2.0, side='right')
        s_bins = np.array([p_rec[a:b].sum() if b > a else 0.0
                           for a, b in zip(w_lo, w_hi)])
        if s_bins.sum() <= 0:
            continue
        s_bins = s_bins / s_bins.sum()
        p_ch = p_tot * sh
        rms_codes = 32768.0 * math.sqrt(p_ch / 0.5) * 0.5
        m = np.sqrt(max(rms_codes, 0.0) ** 2 * s_bins * n_syn * n_syn / 2.0)
        m_full = np.zeros(half)
        m_full[j0:j1 + 1] = m
        spec = m_full * np.exp(1j * rng.uniform(0, 2 * np.pi, half))
        spec[0] = 0.0
        spec[-1] = 0.0
        y = np.fft.irfft(spec, n_syn)
        pk = float(np.abs(y).max())
        if pk > 32000.0:
            y *= 32000.0 / pk
        slot = idx % nch
        cols[slot] = y if cols[slot] is None else cols[slot] + y
    block = np.stack([np.full(n_syn, 32768.0) if c is None else 32768.0 + c
                      for c in cols], axis=1)
    flat = np.clip(np.rint(block), 0, 65535).astype("<u2").reshape(-1)
    return {"flat": flat, "n": n_syn, "nch": nch, "n_sig": len(assign),
            "assign": assign, "p_tot": p_tot}


def analyze_report(path, fs, nch):
    chans, n = load_raw(path, nch)
    dur = n / fs
    fsv, fsu = human_freq(fs)
    lines = ["Файл: %s" % os.path.abspath(path),
             "Відліків на канал: %d   Каналів: %d   fs = %.4g %s   Тривалість: %.4g с"
             % (n, nch, fsv, fsu, dur), ""]
    lines.append("%-6s %10s %9s %8s %8s %9s %9s"
                 % ("канал", "DC", "RMS", "min", "max", "кліп≥65000", "кліп≤500"))
    for k in range(nch):
        st = channel_stats(chans[k], fs)
        lines.append("ch%-4d %10.0f %9.0f %8.0f %8.0f %8.2f%% %7.2f%%"
                     % (k, st["dc"], st["rms"], st["min"], st["max"],
                        st["clip_hi"], st["clip_lo"]))
    lines.append("")
    for k in range(nch):
        f, a = amplitude_spectrum(chans[k], fs)
        pk = find_peaks(f, a, k=5)
        lines.append("ch%d — топ-5 піків спектра:" % k)
        for fq, am in pk:
            fv, fu = human_freq(fq)
            lines.append("    %12.4f %s  ×  %7.1f кодів" % (fv, fu, am))
    return "\n".join(lines), chans, n


# ----------------------------------------------------------------------------
# МАЛЮВАННЯ (спільне для GUI і preview)
# ----------------------------------------------------------------------------
def draw_time(fig, chans, fs, t0, dur, gs=None):
    if gs is not None:
        ax = fig.add_subplot(gs)
    else:
        ax = fig.subplots()
    n_tot = len(chans[0])
    i0 = int(min(max(t0 * fs, 0), n_tot - 1))
    i1 = int(min((t0 + dur) * fs, n_tot))
    i1 = max(i1, i0 + 2)
    dec = max(1, (i1 - i0) // 4000)
    for k in sorted(chans):
        s = chans[k][i0:i1]
        m = (s.size // dec) * dec
        b = s[:m].reshape(-1, dec)
        tt = (i0 + np.arange(b.shape[0]) * dec + dec / 2) / fs
        ax.fill_between(tt, b.min(axis=1), b.max(axis=1), alpha=0.25, color="C%d" % k)
        ax.plot(tt, b.mean(axis=1), lw=0.7, color="C%d" % k, label="ch%d" % k)
    nch = len(chans)
    ax.set_xlabel("час, с"); ax.set_ylabel("коди АЦП")
    ax.set_title("Часова область (%d канали): смуга min–max + середнє (декація 1:%d)"
                 % (nch, dec))
    ax.grid(alpha=0.3); ax.legend(ncol=min(nch, 8))


def _grid_for(nch):
    if nch <= 4:
        return 2, 2
    if nch <= 8:
        return 2, 4
    cols = 4
    return (nch + cols - 1) // cols, cols


def draw_spectrum(fig, chans, fs, gs=None):
    keys = sorted(chans)
    nch = len(keys)
    rows, cols = _grid_for(nch)
    if gs is not None:
        sub = gs.subgridspec(rows, cols)
        axes = [fig.add_subplot(sub[i // cols, i % cols]) for i in range(nch)]
    else:
        axes = fig.subplots(rows, cols, squeeze=False).ravel().tolist()[:nch]
    scale, unit = freq_unit_for(fs)
    fsv, fsu = human_freq(fs)
    report = []
    for k, ax in zip(keys, axes):
        f, a = amplitude_spectrum(chans[k], fs)
        ax.semilogy(f[1:] / scale, a[1:], lw=0.3, color="C%d" % (k % 10))
        ax.set_title("ch%d" % k, color="C%d" % (k % 10))
        ax.set_xlabel("частота, %s (fs = %.4g %s)" % (unit, fsv, fsu))
        ax.set_ylabel("амплітуда, кодів (пік)")
        ax.grid(alpha=0.3, which="both")
        pk = find_peaks(f, a, k=3)
        for fq, am in pk:
            fv, fu = human_freq(fq)
            ax.annotate("%.4g %s" % (fv, fu), (fq / scale, am),
                        fontsize=7, rotation=90, va="bottom")
            report.append("ch%d: %12.4f %s × %7.1f кодів" % (k, fv, fu, am))
    return report


def draw_blocks(fig, chans, fs, blocks, occ, src_key=0):
    f, a = amplitude_spectrum(chans[src_key], fs)
    info = []
    for i, b in enumerate(blocks):
        ax = fig.add_subplot(len(blocks), 1, i + 1)
        m = (f >= b["f0"] - 2e6) & (f <= b["f1"] + 2e6)
        ax.semilogy(f[m] / 1e6, a[m], lw=0.25, color="C%d" % i)
        n_occ = 0
        for ch in b["channels"]:
            if occ[ch["global"]]["occupied"]:
                ax.axvspan(ch["f0"] / 1e6, ch["f1"] / 1e6, color="red", alpha=0.18)
                n_occ += 1
            else:
                ax.axvline(ch["f0"] / 1e6, color="gray", lw=0.3, alpha=0.6)
        ax.axvline(b["f1"] / 1e6, color="gray", lw=0.3, alpha=0.6)
        nb = len(b["channels"])
        npart = sum(1 for ch in b["channels"] if ch["partial"])
        ax.set_title("Блок %d: %.0f–%.0f МГц — %d каналів по 64 МГц%s, зайнято %d"
                     % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6, nb,
                        " (+%d крайовий)" % npart if npart else "", n_occ),
                     fontsize=9, color="C%d" % i)
        ax.set_ylabel("коди", fontsize=7)
        ax.grid(alpha=0.25, which="both")
        if i == len(blocks) - 1:
            ax.set_xlabel("частота, МГц")
        info.append("Блок %d: каналів %d, зайнято %d" % (b["idx"], nb, n_occ))
    return info


def draw_achx_curve(fig, points, title, color="C3", mode="linear", extrap="clamp"):
    ax = fig.subplots()
    fmax = min(1.2 * SIG_MAX_HZ, 12e9)
    fr = np.linspace(SIG_MIN_HZ * 0.5, fmax, 2000)
    ax.plot(fr / 1e9, achx_response(fr, points, mode=mode, extrap=extrap),
            lw=1.3, color=color)
    ax.set_xlabel("частота, ГГц"); ax.set_ylabel("|H(f)|, дБ")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    return ax


def draw_cal_result(fig, measured, applied=None):
    ax = fig.subplots()
    fm = np.array([m[0] for m in measured]) / 1e9
    dm = np.array([m[2] for m in measured])
    ax.plot(fm, dm, "o-", ms=4, lw=0.8, color="C0", label="виміряно з запису")
    if applied:
        fr = np.linspace(min(fm), max(fm), 1000)
        db = np.interp(fr, [p[0] for p in applied], [p[1] for p in applied])
        ax.plot(fr / 1e9, db, lw=1.2, color="C3", label="еталонний профіль тракту")
        ax.legend()
    ax.set_xlabel("частота, ГГц"); ax.set_ylabel("АЧХ тракту, дБ (відн. еталону)")
    ax.set_title("Реальна АЧХ тракту — калібрувальний прогін тонів")
    ax.grid(alpha=0.3)


# ----------------------------------------------------------------------------
# GUI (Tkinter)
# ----------------------------------------------------------------------------
def run_gui(path=None, fs=DEFAULT_FS, nch=DEFAULT_NCH):
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    from matplotlib.figure import Figure
    import matplotlib
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    BLOCKS = build_blocks()
    MODE_KEY = {v: k for k, v in ACHX_MODES.items()}
    EX_KEY = {v: k for k, v in ACHX_EXTRAPS.items()}

    class App(tk.Tk):
        def __init__(self):
            super().__init__()
            self.title(APP_TITLE)
            self.geometry("1400x860")
            self.chans, self.n, self.fs, self.nch = None, 0, fs, nch
            self.last_freq_var = None
            self.measured_achx = None
            top = ttk.Frame(self, padding=6); top.pack(fill="x")
            ttk.Button(top, text="Відкрити .raw…", command=self.open_file).pack(side="left")
            self.lbl_path = ttk.Label(top, text=path or "файл не відкрито", width=52)
            self.lbl_path.pack(side="left", padx=8)
            for txt in ("fs, МГц (можна 12e9):", "каналів:"):
                ttk.Label(top, text=txt).pack(side="left", padx=(10, 2))
            self.fs_var = tk.StringVar(value="%.6g" % (fs / 1e6)); self.nch_var = tk.StringVar(value=str(nch))
            ttk.Entry(top, textvariable=self.fs_var, width=10).pack(side="left")
            ttk.Entry(top, textvariable=self.nch_var, width=4).pack(side="left")
            ttk.Button(top, text="Аналізувати", command=self.analyze).pack(side="left", padx=8)
            self.status = tk.StringVar(value="готовий")
            ttk.Label(self, textvariable=self.status, relief="sunken",
                      anchor="w").pack(side="bottom", fill="x")
            self._help_open = False
            self.btn_help = ttk.Button(self, text="📖 Довідка по параметрах — що означає кожне поле",
                                       command=self._toggle_help)
            self.btn_help.pack(side="bottom", fill="x", padx=6, pady=(2, 0))
            self.txt_help = tk.Text(self, height=10, wrap="word", font=("DejaVu Sans", 9),
                                   state="normal")
            self.txt_help.insert("1.0", self.HELP_TEXT)
            self.txt_help.config(state="disabled")

            nb = ttk.Notebook(self); nb.pack(fill="both", expand=True)
            self.nb = nb
            f1 = ttk.Frame(nb); nb.add(f1, text="Часова область")
            ctl = ttk.Frame(f1); ctl.pack(fill="x", pady=2)
            ttk.Label(ctl, text="з, с:").pack(side="left")
            self.t0_var = tk.StringVar(value="0"); ttk.Entry(ctl, textvariable=self.t0_var, width=8).pack(side="left")
            ttk.Label(ctl, text="тривалість, с:").pack(side="left", padx=(8, 0))
            self.dur_var = tk.StringVar(value="1e-5"); ttk.Entry(ctl, textvariable=self.dur_var, width=8).pack(side="left")
            ttk.Button(ctl, text="Показати", command=self.redraw_time).pack(side="left", padx=8)
            self.fig_t = Figure(figsize=(13, 6), dpi=100)
            self.canvas_t = FigureCanvasTkAgg(self.fig_t, master=f1)
            NavigationToolbar2Tk(self.canvas_t, f1)
            self.canvas_t.get_tk_widget().pack(fill="both", expand=True)

            f2 = ttk.Frame(nb); nb.add(f2, text="Спектр")
            self.fig_s = Figure(figsize=(15.5, 6.2), dpi=100)
            self.canvas_s = FigureCanvasTkAgg(self.fig_s, master=f2)
            NavigationToolbar2Tk(self.canvas_s, f2)
            self.canvas_s.get_tk_widget().pack(fill="both", expand=True)

            f2b = ttk.Frame(nb); nb.add(f2b, text="Блоки (64 МГц)")
            self.fig_b = Figure(figsize=(13, 7.5), dpi=100)
            self.canvas_b = FigureCanvasTkAgg(self.fig_b, master=f2b)
            NavigationToolbar2Tk(self.canvas_b, f2b)
            self.canvas_b.get_tk_widget().pack(fill="both", expand=True)

            f2c = ttk.Frame(nb); nb.add(f2c, text="Калібрування АЧХ")
            self._build_calibration(f2c)

            f3 = ttk.Frame(nb); nb.add(f3, text="Статистика")
            self.txt = tk.Text(f3, wrap="none", font=("DejaVu Sans Mono", 10))
            yb = ttk.Scrollbar(f3, command=self.txt.yview)
            self.txt.configure(yscrollcommand=yb.set)
            yb.pack(side="right", fill="y"); self.txt.pack(fill="both", expand=True)

            self.gen_frame = ttk.Frame(nb); nb.add(self.gen_frame, text="Генератор")
            self._build_generator(self.gen_frame)

            if path and os.path.exists(path):
                self.after(200, lambda: self._load(path))

            fiq = ttk.Frame(nb); nb.add(fiq, text="I/Q → .raw")
            self.iq_frame = fiq
            self._build_iq(fiq)

        HELP_TEXT = (
            "ПАРАМЕТРИ ТАБЛИЦІ ГЕНЕРАТОРА (по рядках каналів)\n"
            "  • Канал (ch0…ch7) — номер канала у файлі .raw; канали чергуються ПО ВІДЛІКАХ:\n"
            "      [ch0, ch1, …, chN-1, ch0, …].\n"
            "  • Увімк — ✓ канал пишеться у файл зі своїм сигналом; ☐ — канал лишається\n"
            "      постійною складовою DC.\n"
            "  • Форма — форма сигналу: Синус / Меандр (прямокутні коливання) / Трикутник /\n"
            "      Імпульс (прямокутні імпульси тривалістю «Ширина») / Шум (білий гаусів;\n"
            "      частота для нього не використовується).\n"
            "  • Частота, МГц — частота тону (100…6000 МГц). Для Меандру/Імпульсу це частота\n"
            "      слідкування (період повторення імпульсів); для Шуму — не використовується (0).\n"
            "      Голе число — МГц; старий запис (2.4e9) і суфікси (2.4ггц) теж розуміються.\n"
            "  • Фаза, ° — початкова фаза тону у градусах (0…359).\n"
            "  • Ампл., кодів — амплітуда сигналу у кодах АЦП: пік ± від DC (10000 ≈ ±10000\n"
            "      кодів від середини шкали; надто велика амплітуда → переповнення, кліп).\n"
            "  • Ширина, мкс — тривалість імпульсу/високого рівня для Імпульсу і Меандру;\n"
            "      мінімально розумна — 4 відліки при fs моделі SDR (кнопка підстановки).\n"
            "  • DC, кодів — постійна складова, середина шкали (32768; сигнал «сидить» на ній).\n"
            "  • Підсилення — множник форми сигналу (1.0 = як є; 2.0 — амплітуда вдвічі більша).\n"
            "\n"
            "ПАРАМЕТРИ ГЕНЕРАЦІЇ (нижній рядок вкладки)\n"
            "  • Блок / Канал — вибір каналу 64 МГц у частотних блоках (100 МГц…6 ГГц); кнопка\n"
            "      «→ центр каналу» підставляє його центральну частоту у виділене поле Частота.\n"
            "  • fs, МГц — частота дискретизації (12 ГГц = пряма оцифровка 6 ГГц; кнопка\n"
            "      «Модель SDR» підставляє максимум обраної моделі).\n"
            "  • тривалість, с — тривалість запису; відліків на канал = тривалість × fs.\n"
            "  • seed — число генератора випадкових чисел: той самий seed → той самий шум.\n"
            "\n"
            "МОДЕЛЬ SDR\n"
            "  • Вибір моделі підставляє fs приладу і показує смугу/діапазон/біти; кнопка\n"
            "      «→ мінімальну ширину» підставляє 4 відліки при fs моделі у поле Ширина;\n"
            "      надто вузькі імпульси (Меандр/Імпульс) — попередження при генерації.\n"
            "\n"
            "ДОДАТКОВІ ПАРАМЕТРИ СИГНАЛУ (рамка під SDR)\n"
            "  • період слідування, мкс — період повторення імпульсів (0 = за Частотою, 1/f).\n"
            "  • тривалість імпульсу, мкс — перевизначає «Ширина» для Імпульсу/Меандру (0 = ні).\n"
            "  • фронт/спад, мкс — швидкість зростання/спаду фронтів (0 = ідеальні стрибки);\n"
            "      Імпульс стає РАДІОімпульсом: несуча (Частота) × огинаюча з фронтами.\n"
            "  • індекс АМ (0…1) + частота АМ, Гц — амплітудна модуляція: огинаюча\n"
            "      (1+m·sin(2π·f_AM·t)); m=0.5 → глибина ±50%.\n"
            "  • девіація ЛЧМ, МГц — лінійна зміна частоти несучого коливання: для Імпульсу\n"
            "      — всередині кожного імпульсу (від f до f+девіація), для Синуса — за весь запис.\n"
            "\n"
            "ТИПИ СИГНАЛІВ (рядок «Тип сигналу:» у Генераторі)\n"
            "Оберіть стандарт — LoRa (868/VHF 150–300/433), LoRaWAN, 2G, 3G, 4G, 5G, WiFi, GNSS, GPS L1, UNB,\n"
            "FSK/GFSK, FHSS, QAM, CVBS, COFDM, NB-IoT — і натисніть «→ застосувати»:\n"
            "програма підставить форму сигналу, несучу (МГц), ширину, fs і всі додаткові\n"
            "параметри (період слідування, тривалість, фронти, АМ, ЛЧМ) у перший УВІМКНЕНИЙ\n"
            "рядок таблиці (якщо увімкнених немає — у рядок 1, тоді увімкніть його вручну).\n"
            "Пакетні стандарти (GSM, LoRaWAN) стають радіоімпульсами; широкосмугові (3G,\n"
            "4G, 5G, WiFi, GNSS, FHSS, QAM, COFDM) — моделлю шумового заповнення канала;\n"
            "CVBS — трапецією в базовій смузі (несуча 0); вузькосмугові (LoRa, UNB, FSK,\n"
            "NB-IoT) — несучою з ЛЧМ/без. Значення — типові, з відкритих специфікацій;\n"
            "коротка модель показана біля списку, повна — у статусному рядку після застосування.\n"
            "\n"
            "КОНВЕРТЕР I/Q → .raw (вкладка «I/Q → .raw»)\n"
            "Файл I/Q — запис RTSA-Suite: .iq (Raw IQ, інтерлівінг Re,Im), .wav (стерео)\n"
            "або будь-який бінарний raw; тип даних float32/float64/int8/int16/int32.\n"
            "Якщо поруч лежить .iq.xml (RTSA_RAW_IQ_File), fs і центральна частота\n"
            "підставляться автоматично; інакше введіть їх вручну (поля в МГц).\n"
            "Нативні стиснуті .rtsa не читаються — експортуйте Raw IQ у File Source.\n"
            "Конвертор будує середній спектр запису (Уелч, до 200 сегментів, перші ~4 млн\n"
            "відліків), розкладає потужність по каналах плану 64 МГц і синтезує для\n"
            "кожного канала часовий ряд із цим профілем: піки, рівні в кодах і зайнятість\n"
            "каналів у RAW Studio відповідають запису; порожні канали — DC 32768.\n"
            "Повна шкала запису (комплексна амплітуда 1.0) = несуча 32768 кодів; запас\n"
            "−6 дБ, без кліпінгу. fs .raw (МГц) — те, що треба вказати при відкритті файлу.\n"
            "\n"
            "МОДУЛЬ АЧХ (під рядком SDR)\n"
            "  • застосовувати АЧХ — форма змінної складової залежно від частоти за профілем;\n"
            "      постійна складова DC не модулюється.\n"
            "  • профіль — Плоский / Спад −3 дБ/окт / Смуговий 100М–6Г / Провал / Користувацький.\n"
            "  • інтерполяція — linear / spline / logspline (між точками профілю); поза діапазоном\n"
            "      — clamp / linear / zero (за межами крайніх точок таблиці).\n"
            "  • центр провалу, МГц — частота провалу для профілю «Провал».\n"
            "  • власна таблиця (частота_Гц дБ) — рядки профілю; сюди потрапляє виміряна АЧХ тракту.\n"
            "\n"
            "КАЛІБРУВАННЯ АЧХ (окрема вкладка)\n"
            "  • fs, каналів, тонів — параметри прогону; від/до, МГц — діапазон тонів (кожен тон\n"
            "      точно в біні FFT); ампл. тону — амплітуда еталона; відліків/тон — довжина сегмента.\n"
            "  • Крок 1 — створити калібрувальний .raw → прогнати через тракт → записати;\n"
            "      крок 2 — виміряти АЧХ із запису; крок 3 — профіль у Генератор (компенсація −дБ);\n"
            "      💾 / 📁 — збереження/завантаження профілю .achx (файли у Гц, режими — в шапці).\n"
            "\n"
            "ВЕРХНЄ ПОЛЕ (аналіз)\n"
            "  • fs, МГц — частота дискретизації ВІДКРИТОГО файла; каналів — кількість каналів;\n"
            "      кнопка «Аналізувати» — перечитати файл і оновити всі вкладки."
        )

        def _show_sig(self):
            pr = SIG_PRESETS.get(self.cb_sig.get())
            if pr:
                self.lbl_sig.config(text=pr["label"])

        def _apply_sig(self):
            name = self.cb_sig.get()
            pr = SIG_PRESETS.get(name)
            if not pr:
                return
            target = None
            for en, kind, vals in self.gen_rows:
                if bool(en.get()):
                    target = (en, kind, vals)
                    break
            if target is None:
                target = self.gen_rows[0]
                self._set_status("тип сигналу: увімкнених рядків немає — параметри записані в рядок 1 (увімкніть його)")
            en, kind, vals = target
            kind.set(pr["kind"])
            vals[0].set("%.6g" % (pr["freq"] / 1e6))
            if pr.get("width_us"):
                vals[3].set("%.6g" % pr["width_us"])
            ex = pr["extra"]
            self.ex_per.set("%.6g" % ex.get("per_us", 0))
            self.ex_pw.set("%.6g" % ex.get("pw_us", 0))
            self.ex_rise.set("%.6g" % ex.get("rise_us", 0))
            self.ex_fall.set("%.6g" % ex.get("fall_us", 0))
            self.ex_am_m.set("%.6g" % ex.get("am_m", 0))
            self.ex_am_f.set("%.6g" % ex.get("am_f", 0))
            self.ex_chirp.set("%.6g" % (ex.get("chirp_dev", 0) / 1e6))
            self.gen_fs.set("%.6g" % (pr["fs_hint"] / 1e6))
            self._set_status("тип сигналу «%s» → %s: %s" % (name, pr["kind"], pr["note"]))

        def _build_iq(self, fiq):
            r1 = ttk.Frame(fiq); r1.pack(fill="x", pady=(8, 2))
            ttk.Label(r1, text="Файл I/Q (RTSA-Suite: .iq + .iq.xml, .wav, raw):").pack(side="left")
            self.iq_path = tk.StringVar(value="")
            ttk.Entry(r1, textvariable=self.iq_path, width=42).pack(side="left", padx=4)
            ttk.Button(r1, text="Огляд…", command=self._iq_pick).pack(side="left")
            r2 = ttk.Frame(fiq); r2.pack(fill="x", pady=2)
            ttk.Label(r2, text="тип даних:").pack(side="left")
            self.iq_dtype = ttk.Combobox(r2, values=list(RTSA_IQ_DTYPES.keys()),
                                         width=9, state="readonly")
            self.iq_dtype.current(0); self.iq_dtype.pack(side="left", padx=3)
            ttk.Label(r2, text="fs I/Q, МС/с:").pack(side="left", padx=(10, 0))
            self.iq_fs = tk.StringVar(value="")
            ttk.Entry(r2, textvariable=self.iq_fs, width=10).pack(side="left", padx=3)
            ttk.Label(r2, text="центр, МГц:").pack(side="left", padx=(10, 0))
            self.iq_fc = tk.StringVar(value="")
            ttk.Entry(r2, textvariable=self.iq_fc, width=10).pack(side="left", padx=3)
            r3 = ttk.Frame(fiq); r3.pack(fill="x", pady=2)
            ttk.Label(r3, text="довжина канала .raw, відліків:").pack(side="left")
            self.iq_n = tk.StringVar(value="16384")
            ttk.Entry(r3, textvariable=self.iq_n, width=9).pack(side="left", padx=3)
            ttk.Label(r3, text="каналів у .raw:").pack(side="left", padx=(10, 0))
            self.iq_nch = tk.StringVar(value="8")
            ttk.Entry(r3, textvariable=self.iq_nch, width=5).pack(side="left", padx=3)
            ttk.Label(r3, text="fs .raw, МГц (або 12e9):").pack(side="left", padx=(10, 0))
            self.iq_fsout = tk.StringVar(value="12000")
            ttk.Entry(r3, textvariable=self.iq_fsout, width=10).pack(side="left", padx=3)
            r4 = ttk.Frame(fiq); r4.pack(fill="x", pady=(6, 2))
            ttk.Button(r4, text="Конвертувати → .raw…", command=self._iq_convert).pack(side="left")
            ttk.Label(fiq, text="Запис RTSA-Suite: експортуйте Raw IQ (.iq) з File Source; "
                                "центр і fs підуть із сусіднього .iq.xml, інакше введіть вручну "
                                "(поля в МГц). Нативні стиснуті .rtsa не читаються. Конвертор "
                                "розкладає середній спектр запису по каналах плану 64 МГц: "
                                "піки, рівні в кодах і зайнятість каналів відтворюються; "
                                "при відкритті .raw вкажіть ту саму fs.",
                      wraplength=780, justify="left").pack(fill="x", pady=(6, 2))

        def _iq_pick(self):
            p = filedialog.askopenfilename(
                title="I/Q запис RTSA-Suite",
                filetypes=[("I/Q записи", "*.iq *.wav *.dat *.raw *.f32"),
                           ("Усі файли", "*.*")])
            if not p:
                return
            self.iq_path.set(p)
            info = parse_rtsa_xml(p + ".xml")
            if info is None:
                info = parse_rtsa_xml(os.path.splitext(p)[0] + ".xml")
            if info:
                if info.get("fs_iq"):
                    self.iq_fs.set("%.6g" % (info["fs_iq"] / 1e6))
                if info.get("fc") is not None:
                    self.iq_fc.set("%.6g" % (info["fc"] / 1e6))
                if info.get("dtype") in RTSA_IQ_DTYPES:
                    self.iq_dtype.set(info["dtype"])
                self._set_status("I/Q: метадані взято з .iq.xml")
            else:
                self._set_status("I/Q: .iq.xml не знайдено — введіть fs і центр вручну")

        def _iq_convert(self):
            p = self.iq_path.get().strip()
            if not p or not os.path.exists(p):
                messagebox.showerror(APP_TITLE, "Вкажіть існуючий файл I/Q")
                return
            try:
                fs_iq = parse_freq_hz(self.iq_fs.get())
                fc = parse_freq_hz(self.iq_fc.get())
                fs_out = parse_freq_hz(self.iq_fsout.get())
                n_per_ch = int(float(self.iq_n.get()))
                nch = int(float(self.iq_nch.get()))
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні числа в полях конвертера I/Q")
                return
            try:
                x, meta = read_iq(p, dtype=self.iq_dtype.get(), fs_iq=fs_iq, fc=fc)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "I/Q: %s" % e)
                return
            out = filedialog.asksaveasfilename(
                initialdir=os.path.dirname(p) or os.getcwd(),
                defaultextension=".raw",
                initialfile=os.path.splitext(os.path.basename(p))[0] + ".raw",
                title="Куди зберегти .raw", filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("конвертація I/Q → .raw…")
            self.update()
            try:
                info = iq_to_raw(x, fc, fs_iq, fs_out=fs_out,
                                 n_per_ch=n_per_ch, nch=nch)
                save_raw(out, info["flat"])
            except Exception as e:
                messagebox.showerror(APP_TITLE, "I/Q: %s" % e)
                return
            self._set_status(
                "I/Q → .raw: %s — каналів плану з сигналом: %d, %d відліків/канал, "
                "nch=%d, %.1f МБ; при відкритті вкажіть fs %.4g"
                % (os.path.basename(out), info["n_sig"], info["n"], nch,
                   os.path.getsize(out) / 1e6, fs_out))

        def _toggle_help(self):
            if self._help_open:
                self.txt_help.pack_forget()
                self.geometry("%dx%d" % (self.winfo_width(),
                                         max(600, self.winfo_height() - self.txt_help.winfo_reqheight() - 90)))
                self.btn_help.config(text="📖 Довідка по параметрах — що означає кожне поле")
                self._help_open = False
            else:
                self.txt_help.pack(side="bottom", fill="both")
                self.geometry("%dx%d" % (self.winfo_width(),
                                         self.winfo_height() + self.txt_help.winfo_reqheight() + 90))
                self.btn_help.config(text="✖ Згорнути довідку")
                self._help_open = True
            self._set_status("довідка по параметрах: %s"
                             % ("розгорнуто" if self._help_open else "згорнуто"))

        def _set_status(self, s):
            self.status.set(s); self.update_idletasks()

        def open_file(self):
            p = filedialog.askopenfilename(title="Виберіть .raw файл",
                                           filetypes=[("RAW файли", "*.raw"), ("Усі файли", "*.*")])
            if p:
                self._load(p)

        def _load(self, p):
            try:
                self.fs = parse_freq_hz(self.fs_var.get())
                self.nch = int(self.nch_var.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні fs або кількість каналів"); return
            self._set_status("читаю %s …" % os.path.basename(p))
            try:
                self.chans, self.n = load_raw(p, self.nch)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Не вдалося прочитати: %s" % e); return
            self.path = p
            self.lbl_path.config(text=p)
            if len(self.gen_rows) != self.nch:
                for w in self.gen_frame.winfo_children():
                    w.destroy()
                self._build_generator(self.gen_frame)
            self.analyze()

        def analyze(self):
            if self.chans is None:
                return
            self._set_status("аналізую…")
            self.fig_t.clear(); self.fig_s.clear(); self.fig_b.clear()
            draw_time(self.fig_t, self.chans, self.fs, 0.0, min(1e-5, self.n / self.fs))
            draw_spectrum(self.fig_s, self.chans, self.fs)
            self.occ = channel_occupancy(self.chans, self.fs, BLOCKS)
            info = draw_blocks(self.fig_b, self.chans, self.fs, BLOCKS, self.occ)
            self.canvas_t.draw_idle(); self.canvas_s.draw_idle(); self.canvas_b.draw_idle()
            report, _, _ = analyze_report(self.path, self.fs, self.nch)
            report += "\n\n--- Зайнятість 64-МГц каналів по блоках ---\n" + "\n".join(info)
            fc_of = {c["global"]: c["fc"] for b in BLOCKS for c in b["channels"]}
            occ_list = [gg for gg in sorted(self.occ) if self.occ[gg]["occupied"]]
            if occ_list:
                report += "\nЗайняті канали: " + ", ".join(
                    "К%03d(%.0f МГц)" % (gg, fc_of[gg] / 1e6) for gg in occ_list)
            else:
                report += "\nЗайняті канали: немає"
            self.txt.delete("1.0", "end"); self.txt.insert("1.0", report)
            fv, fu = human_freq(self.n / self.fs)
            self._set_status("файл: %s — %d відліків/канал, %.4g %s"
                             % (os.path.basename(self.path), self.n, fv, fu))

        def redraw_time(self):
            if self.chans is None:
                return
            try:
                t0 = float(self.t0_var.get()); dur = float(self.dur_var.get())
            except ValueError:
                return
            self.fig_t.clear()
            draw_time(self.fig_t, self.chans, self.fs, t0, dur)
            self.canvas_t.draw_idle()

        # ---------- калібрування АЧХ ----------
        def _build_calibration(self, parent):
            fr = ttk.Frame(parent, padding=8); fr.pack(fill="x")
            self.cal_unit = tk.StringVar(value="МГц")
            ttk.Button(fr, text="Одиниці: МГц ⇄ Гц", width=18,
                       command=self._toggle_cal_unit).pack(side="left", padx=(0, 10))
            self.lbl_cal_fs = ttk.Label(fr, text="fs, МГц (або 12e9):"); self.lbl_cal_fs.pack(side="left")
            self.cal_fs = tk.StringVar(value="%.6g" % (DEFAULT_FS / 1e6))
            ttk.Entry(fr, textvariable=self.cal_fs, width=11).pack(side="left", padx=3)
            ttk.Label(fr, text="каналів:").pack(side="left")
            self.cal_nch = tk.StringVar(value=str(DEFAULT_NCH))
            ttk.Entry(fr, textvariable=self.cal_nch, width=4).pack(side="left", padx=3)
            ttk.Label(fr, text="тонів:").pack(side="left")
            self.cal_tones = tk.StringVar(value="32")
            ttk.Entry(fr, textvariable=self.cal_tones, width=5).pack(side="left", padx=3)
            self.lbl_cal_fmin = ttk.Label(fr, text="від, МГц:"); self.lbl_cal_fmin.pack(side="left")
            self.cal_fmin = tk.StringVar(value="100")
            ttk.Entry(fr, textvariable=self.cal_fmin, width=9).pack(side="left", padx=3)
            self.lbl_cal_fmax = ttk.Label(fr, text="до, МГц:"); self.lbl_cal_fmax.pack(side="left")
            self.cal_fmax = tk.StringVar(value="6000")
            ttk.Entry(fr, textvariable=self.cal_fmax, width=9).pack(side="left", padx=3)
            ttk.Label(fr, text="ампл. тону, кодів:").pack(side="left")
            self.cal_amp = tk.StringVar(value="8000")
            ttk.Entry(fr, textvariable=self.cal_amp, width=7).pack(side="left", padx=3)
            ttk.Label(fr, text="відліків/тон:").pack(side="left")
            self.cal_seg = tk.StringVar(value="16384")
            ttk.Entry(fr, textvariable=self.cal_seg, width=8).pack(side="left", padx=3)
            row2 = ttk.Frame(parent, padding=(8, 0)); row2.pack(fill="x")
            ttk.Button(row2, text="1) Створити калібрувальний .raw…",
                       command=self.cal_generate).pack(side="left")
            ttk.Button(row2, text="2) Виміряти АЧХ із записаного файла…",
                       command=self.cal_measure).pack(side="left", padx=8)
            ttk.Button(row2, text="3) → Профіль у Генератор (компенсація −дБ / як виміряно)",
                       command=self.cal_to_profile).pack(side="left")
            row3 = ttk.Frame(parent, padding=(8, 4)); row3.pack(fill="x")
            ttk.Button(row3, text="💾 Зберегти виміряний профіль АЧХ у файл…",
                       command=self.cal_save_profile).pack(side="left")
            ttk.Button(row3, text="📁 Завантажити профіль АЧХ з файлу → Генератор",
                       command=self.cal_load_profile).pack(side="left", padx=8)
            self.fig_c = Figure(figsize=(12.5, 4.6), dpi=100)
            self.canvas_c = FigureCanvasTkAgg(self.fig_c, master=parent)
            self.canvas_c.get_tk_widget().pack(fill="both", expand=True, padx=8, pady=4)
            self.txt_cal = tk.Text(parent, height=9, font=("DejaVu Sans Mono", 9))
            self.txt_cal.pack(fill="x", padx=8, pady=(0, 8))
            ttk.Label(parent, foreground="#555", wraplength=1250, justify="left",
                      text="Методика: створіть калібрувальний файл (ступінчастий прогін тонів, "
                           "кожен тон точно в біні FFT), проганіть через реальний тракт і запишіть "
                           "відповідь у .raw, виміряйте АЧХ. Профіль зберігається у текстових .achx "
                           "(рядки «частота_Гц дБ», # — коментарі; режим інтерполяції/екстраполяції — у шапці) "
                           "і завантажується назад у Генератор однією кнопкою. "
                           "Між точками профіль інтерполюється (лінійно/сплайном) — працює на будь-якій частоті. "
                           "Одиниці частот перемикаються кнопкою «Одиниці: МГц ⇄ Гц»; .plan- і .achx-файли лишаються у Гц."
                      ).pack(anchor="w", padx=8, pady=2)

        def _toggle_cal_unit(self):
            cur = self.cal_unit.get()
            for var in (self.cal_fs, self.cal_fmin, self.cal_fmax):
                try:
                    hz = parse_freq_hz(var.get())
                except ValueError:
                    continue
                var.set("%.10g" % (hz if cur == "МГц" else hz / 1e6))
            if cur == "МГц":
                self.cal_unit.set("Гц")
                self.lbl_cal_fs.config(text="fs, Гц:")
                self.lbl_cal_fmin.config(text="від, Гц:")
                self.lbl_cal_fmax.config(text="до, Гц:")
            else:
                self.cal_unit.set("МГц")
                self.lbl_cal_fs.config(text="fs, МГц (або 12e9):")
                self.lbl_cal_fmin.config(text="від, МГц:")
                self.lbl_cal_fmax.config(text="до, МГц:")
            self._set_status("вкладка «Калібрування АЧХ»: одиниці частот — %s" % self.cal_unit.get())

        def _cal_params(self):
            return (parse_freq_hz(self.cal_fs.get()), int(self.cal_nch.get()),
                    int(self.cal_tones.get()), parse_freq_hz(self.cal_fmin.get()),
                    parse_freq_hz(self.cal_fmax.get()), float(self.cal_amp.get()),
                    int(self.cal_seg.get()))

        def cal_generate(self):
            try:
                fs_c, nch_c, tones, fmin, fmax, amp, seg_n = self._cal_params()
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні параметри калібрування"); return
            freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
            out = filedialog.asksaveasfilename(defaultextension=".raw",
                                               initialfile="calibration_sweep.raw",
                                               title="Куди зберегти калібрувальний .raw",
                                               filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("генерую калібрування: %d тонів…" % len(freqs))
            save_raw(out, generate_cal_file(fs_c, nch_c, freqs, amp, seg_n))
            save_cal_plan(out + ".plan", fs_c, nch_c, freqs, amp, seg_n)
            self._set_status("калібрувальний файл: %s (+ .plan)" % os.path.basename(out))
            self.txt_cal.delete("1.0", "end")
            self.txt_cal.insert("1.0", "План тонів (бін FFT = %.6g МГц):\n%s"
                                % (bin_hz / 1e6, "\n".join("  %.6g МГц" % (fq / 1e6) for fq in freqs)))

        def cal_measure(self):
            rec = filedialog.askopenfilename(title="Запис після тракту (.raw)",
                                             filetypes=[("RAW файли", "*.raw")])
            if not rec:
                return
            plan_p = filedialog.askopenfilename(title="Файл плану (.raw.plan)",
                                                filetypes=[("Plan", "*.plan"), ("Усі файли", "*.*")])
            if not plan_p:
                return
            try:
                plan = load_cal_plan(plan_p)
                res = measure_achx(rec, plan)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Помилка вимірювання: %s" % e); return
            self.measured_achx = [(fq, gdb) for fq, _, gdb in res]
            self.fig_c.clear()
            draw_cal_result(self.fig_c, res)
            self.canvas_c.draw_idle()
            self.txt_cal.delete("1.0", "end")
            self.txt_cal.insert("1.0", "Реальна АЧХ тракту (частота, ампл. кодів, дБ відн. еталону %d):\n%s"
                                % (plan["amp"], "\n".join("  %12.6g МГц  %8.1f кодів  %+6.2f дБ" % (fq / 1e6, amp, gdb)
                                                         for fq, amp, gdb in res)))

        def cal_to_profile(self):
            if not self.measured_achx:
                messagebox.showinfo(APP_TITLE, "Спочатку виміряйте АЧХ (крок 2)"); return
            compensate = messagebox.askyesno(
                APP_TITLE, "Так — зберегти КОМПЕНСАЦІЮ (−дБ, вирівнює тракт);\n"
                           "Ні — зберегти як виміряно (+дБ, модель тракту)?")
            lines = "\n".join("%.6g %.3f" % (fq, -gdb if compensate else gdb)
                              for fq, gdb in self.measured_achx)
            for w in self.gen_frame.winfo_children():
                w.destroy()
            self._build_generator(self.gen_frame)
            self.cb_achx.set("Користувацький")
            self.achx_on.set(True)
            self.txt_achx.delete("1.0", "end")
            self.txt_achx.insert("1.0", lines)
            self.nb.select(self.gen_frame)
            self._set_status("профіль АЧХ (%s) перенесено в Генератор"
                             % ("компенсація −дБ" if compensate else "як виміряно"))

        def cal_save_profile(self):
            if not self.measured_achx:
                messagebox.showinfo(APP_TITLE, "Спочатку виміряйте АЧХ (крок 2)"); return
            compensate = messagebox.askyesno(
                APP_TITLE, "Так — зберегти КОМПЕНСАЦІЮ (−дБ);\nНі — як виміряно (+дБ)?")
            pts = [(fq, -gdb if compensate else gdb) for fq, gdb in self.measured_achx]
            out = filedialog.asksaveasfilename(defaultextension=".achx",
                                               initialfile="achx_tract.achx",
                                               title="Куди зберегти профіль АЧХ",
                                               filetypes=[("Профіль АЧХ", "*.achx"), ("Текст", "*.txt")])
            if not out:
                return
            save_achx_profile(out, pts, mode=self.achx_mode_key(),
                              extrap=self.achx_extrap_key(), meta={
                                  "fs": self.fs_var.get(),
                                  "компенсація": "так" if compensate else "ні",
                                  "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                                  "source": getattr(self, "path", "-")})
            self._set_status("профіль АЧХ збережено: %s (%d точок)"
                             % (os.path.basename(out), len(pts)))

        def cal_load_profile(self):
            p = filedialog.askopenfilename(title="Файл профілю АЧХ",
                                           filetypes=[("Профіль АЧХ", "*.achx"), ("Текст", "*.txt"), ("Усі файли", "*.*")])
            if not p:
                return
            try:
                pts, mode_b, ex_b = load_achx_profile(p)
            except Exception as e:
                messagebox.showerror(APP_TITLE, "Профіль АЧХ: %s" % e); return
            for w in self.gen_frame.winfo_children():
                w.destroy()
            self._build_generator(self.gen_frame)
            self.cb_achx.set("Користувацький")
            self.achx_on.set(True)
            self.cb_achx_mode.set(MODE_KEY.get(mode_b, "Лінійна"))
            self.cb_achx_ex.set(EX_KEY.get(ex_b, "За краями (clamped)"))
            self.txt_achx.delete("1.0", "end")
            self.txt_achx.insert("1.0", "\n".join("%.6g %.4f" % pt for pt in pts))
            self.nb.select(self.gen_frame)
            fv, fu = human_freq(pts[-1][0])
            self._set_status("профіль АЧХ завантажено: %s — %d точок, до %.4g %s (%s/%s)"
                             % (os.path.basename(p), len(pts), fv, fu, mode_b, ex_b))

        # ---------- генератор ----------
        def _build_generator(self, parent):
            nch_here = self.nch
            g = ttk.Frame(parent, padding=8); g.pack(fill="both", expand=True)
            cols = ["Канал", "Увімк", "Форма", "Частота, МГц (100…6000)", "Фаза, °",
                    "Ампл., кодів", "Ширина, мкс", "DC, кодів", "Підсилення"]
            for j, c in enumerate(cols):
                ttk.Label(g, text=c, font=("DejaVu Sans", 9, "bold")).grid(row=0, column=j, padx=4)
            self.gen_rows = []
            defaults4 = [
                ("Синус", "852", 0, 10000, 2.0, 32768, 1.0),
                ("Синус", "2400", 30, 8000, 2.0, 33000, 1.0),
                ("Шум", "0", 0, 800, 100, 40000, 1.0),
                ("Імпульс", "435", 0, 20000, 100.0, 1000, 1.0),
            ]
            for k in range(nch_here):
                en = tk.BooleanVar(value=(k < 2))
                kind = ttk.Combobox(g, values=WAVE_KINDS, width=10, state="readonly")
                if k < len(defaults4):
                    kind.set(defaults4[k][0])
                    vals = [tk.StringVar(value=str(v)) for v in defaults4[k][1:]]
                else:
                    kind.set("Синус")
                    vals = [tk.StringVar(value=v) for v in ("1000", "0", "5000", "2.0", "32768", "1.0")]
                ttk.Label(g, text="ch%d" % k).grid(row=k + 1, column=0)
                ttk.Checkbutton(g, variable=en).grid(row=k + 1, column=1)
                kind.grid(row=k + 1, column=2, padx=4)
                for j, v in enumerate(vals):
                    e = ttk.Entry(g, textvariable=v, width=10)
                    e.grid(row=k + 1, column=j + 3, padx=3)
                    if j == 0:
                        e.bind("<FocusIn>", lambda ev, v=v: setattr(self, "last_freq_var", v))
                    if j == 3:
                        e.bind("<FocusIn>", lambda ev, v=v: setattr(self, "last_width_var", v))
                self.gen_rows.append((en, kind, vals))
            rgen = nch_here + 1
            gg = ttk.Frame(g); gg.grid(row=rgen, column=0, columnspan=9, sticky="w", pady=(10, 4))
            ttk.Label(gg, text="Блок:").pack(side="left")
            self.cb_block = ttk.Combobox(gg, values=["%d: %.0f–%.0f МГц" % (b["idx"], b["f0"] / 1e6, b["f1"] / 1e6)
                                                     for b in BLOCKS], width=16, state="readonly")
            self.cb_block.current(0); self.cb_block.pack(side="left", padx=3)
            ttk.Label(gg, text="Канал:").pack(side="left")
            self.cb_ch = ttk.Combobox(gg, width=8, state="readonly")
            self.cb_ch.pack(side="left", padx=3)
            self.lbl_fc = ttk.Label(gg, text="")
            self.lbl_fc.pack(side="left", padx=6)
            ttk.Button(gg, text="→ центр каналу у виділене поле частоти",
                       command=self.insert_channel_freq).pack(side="left", padx=8)
            self._fill_channels()
            self.cb_block.bind("<<ComboboxSelected>>", lambda e: self._fill_channels())
            ttk.Label(gg, text="fs, МГц (або 12e9):").pack(side="left")
            self.gen_fs = tk.StringVar(value="%.6g" % (DEFAULT_FS / 1e6))
            ttk.Entry(gg, textvariable=self.gen_fs, width=11).pack(side="left", padx=3)
            ttk.Label(gg, text="тривалість, с:").pack(side="left")
            self.gen_dur = tk.StringVar(value="1e-4")
            ttk.Entry(gg, textvariable=self.gen_dur, width=8).pack(side="left", padx=3)
            ttk.Label(gg, text="seed:").pack(side="left")
            self.gen_seed = tk.StringVar(value="42")
            ttk.Entry(gg, textvariable=self.gen_seed, width=7).pack(side="left", padx=3)
            ttk.Button(gg, text="Створити .raw…", command=self.generate).pack(side="left", padx=12)

            gs = ttk.Frame(g); gs.grid(row=rgen + 1, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(gs, text="Модель SDR:").pack(side="left")
            self.cb_sdr = ttk.Combobox(gs, values=list(SDR_PRESETS.keys()), width=22, state="readonly")
            self.cb_sdr.current(0); self.cb_sdr.pack(side="left", padx=3)
            self.lbl_sdr = ttk.Label(gs, text=""); self.lbl_sdr.pack(side="left", padx=6)
            ttk.Button(gs, text="→ мінімальну ширину у виділене поле ширини",
                       command=self.insert_sdr_width).pack(side="left", padx=8)
            self._show_sdr()
            self.cb_sdr.bind("<<ComboboxSelected>>", lambda e: self._on_sdr_selected())
            rsig = ttk.Frame(g); rsig.grid(row=rgen + 2, column=0, columnspan=9, sticky="w", pady=(6, 0))
            ttk.Label(rsig, text="Тип сигналу:").pack(side="left")
            self.cb_sig = ttk.Combobox(rsig, values=list(SIG_PRESETS.keys()), width=20, state="readonly")
            self.cb_sig.current(0); self.cb_sig.pack(side="left", padx=3)
            self.lbl_sig = ttk.Label(rsig, text=""); self.lbl_sig.pack(side="left", padx=6)
            ttk.Button(rsig, text="→ застосувати до Генератора (форма, частота, fs, додаткові)",
                       command=self._apply_sig).pack(side="left", padx=8)
            self._show_sig()
            self.cb_sig.bind("<<ComboboxSelected>>", lambda e: self._show_sig())


            gx = ttk.LabelFrame(g, text="Додаткові параметри сигналу — період слідування, тривалість імпульсу, фронти, АМ, ЛЧМ")
            gx.grid(row=rgen + 3, column=0, columnspan=9, sticky="we", pady=(6, 2), padx=2)
            gxr1 = ttk.Frame(gx); gxr1.pack(fill="x", pady=1)
            self.ex_per = tk.StringVar(value="0")
            ttk.Label(gxr1, text="період слідування, мкс (0 = 1/Частота):").pack(side="left")
            ttk.Entry(gxr1, textvariable=self.ex_per, width=9).pack(side="left", padx=3)
            self.ex_pw = tk.StringVar(value="0")
            ttk.Label(gxr1, text="тривалість імпульсу, мкс (0 = за «Ширина»):").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_pw, width=9).pack(side="left", padx=3)
            self.ex_rise = tk.StringVar(value="0")
            ttk.Label(gxr1, text="фронт (зростання), мкс:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_rise, width=9).pack(side="left", padx=3)
            self.ex_fall = tk.StringVar(value="0")
            ttk.Label(gxr1, text="спад, мкс:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr1, textvariable=self.ex_fall, width=9).pack(side="left", padx=3)
            gxr2 = ttk.Frame(gx); gxr2.pack(fill="x", pady=1)
            self.ex_am_m = tk.StringVar(value="0")
            ttk.Label(gxr2, text="індекс АМ (0…1, 0 = без):").pack(side="left")
            ttk.Entry(gxr2, textvariable=self.ex_am_m, width=9).pack(side="left", padx=3)
            self.ex_am_f = tk.StringVar(value="0")
            ttk.Label(gxr2, text="частота АМ, Гц:").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr2, textvariable=self.ex_am_f, width=9).pack(side="left", padx=3)
            self.ex_chirp = tk.StringVar(value="0")
            ttk.Label(gxr2, text="девіація ЛЧМ, МГц (0 = без):").pack(side="left", padx=(10, 0))
            ttk.Entry(gxr2, textvariable=self.ex_chirp, width=9).pack(side="left", padx=3)

            ach = ttk.LabelFrame(g, text="Модуль АЧХ АЦП — амплітуда залежно від частоти")
            ach.grid(row=rgen + 4, column=0, columnspan=9, sticky="we", pady=(8, 2), padx=2)
            rowa = ttk.Frame(ach); rowa.pack(fill="x", pady=2)
            self.achx_on = tk.BooleanVar(value=False)
            ttk.Checkbutton(rowa, text="застосовувати АЧХ", variable=self.achx_on).pack(side="left")
            ttk.Label(rowa, text="профіль:").pack(side="left", padx=(12, 2))
            self.cb_achx = ttk.Combobox(rowa, values=ACHX_PRESETS, width=18, state="readonly")
            self.cb_achx.current(0); self.cb_achx.pack(side="left")
            ttk.Label(rowa, text="інтерполяція:").pack(side="left", padx=(12, 2))
            self.cb_achx_mode = ttk.Combobox(rowa, values=list(ACHX_MODES.keys()),
                                             width=15, state="readonly")
            self.cb_achx_mode.set("Кубічний сплайн"); self.cb_achx_mode.pack(side="left")
            ttk.Label(rowa, text="поза діапазоном:").pack(side="left", padx=(12, 2))
            self.cb_achx_ex = ttk.Combobox(rowa, values=list(ACHX_EXTRAPS.keys()),
                                           width=18, state="readonly")
            self.cb_achx_ex.set("За краями (clamped)"); self.cb_achx_ex.pack(side="left")
            rowa2 = ttk.Frame(ach); rowa2.pack(fill="x", pady=2)
            ttk.Label(rowa2, text="центр провалу:").pack(side="left")
            self.achx_notch = tk.StringVar(value="1000")
            ttk.Entry(rowa2, textvariable=self.achx_notch, width=9).pack(side="left", padx=3)
            ttk.Label(rowa2, text="МГц").pack(side="left")
            rowb = ttk.Frame(ach); rowb.pack(fill="x", pady=2)
            ttk.Label(rowb, text="власна таблиця (рядки: частота_Гц дБ) — сюди потрапляє виміряна АЧХ тракту:").pack(anchor="w")
            self.txt_achx = tk.Text(rowb, height=4, width=46, font=("DejaVu Sans Mono", 9))
            self.txt_achx.insert("1.0", "100e6 0\n1e9 -3\n3e9 -6\n6e9 -12")
            self.txt_achx.pack(anchor="w")
            n_full = sum(1 for b in BLOCKS for c in b["channels"] if not c["partial"])
            n_part = sum(1 for b in BLOCKS for c in b["channels"] if c["partial"])
            ttk.Label(g, foreground="#555", wraplength=1250, justify="left",
                      text="Частотний план: 6 блоків, каналізація %.0f МГц: %d повних + %d крайових = %d каналів. "
                           "АЧХ оцінюється на БУДЬ-ЯКІЙ частоті: між точками — інтерполяція "
                           "(лінійна / кубічний сплайн / сплайн у log f), поза виміряним діапазоном — "
                           "за краями / лінійна екстраполяція / нуль. Застосовується до змінної складової "
                           "усіх увімкнених каналів (DC не модулюється). Виміряну АЧХ тракту беріть на "
                           "вкладці «Калібрування АЧХ»; збереження/завантаження — .achx."
                           % (CHANNEL_BW / 1e6, n_full, n_part, n_full + n_part)
                      ).grid(row=rgen + 5, column=0, columnspan=9, sticky="w")

        def achx_mode_key(self):
            return ACHX_MODES.get(self.cb_achx_mode.get(), "linear")

        def achx_extrap_key(self):
            return ACHX_EXTRAPS.get(self.cb_achx_ex.get(), "clamp")

        def _fill_channels(self):
            b = BLOCKS[self.cb_block.current()]
            self.cb_ch["values"] = ["К%03d / %02d (%s)" % (c["global"], c["in_block"],
                                                           ("%.0f" % (c["fc"] / 1e6) + " МГц"))
                                    for c in b["channels"]]
            self.cb_ch.current(0)
            self._show_fc()

        def _show_fc(self):
            b = BLOCKS[self.cb_block.current()]
            ch = b["channels"][self.cb_ch.current()]
            fv, fu = human_freq(ch["fc"])
            self.lbl_fc.config(text="центр: %.4g %s (ширина %.0f МГц%s)"
                               % (fv, fu, ch["w"] / 1e6, ", крайовий" if ch["partial"] else ""))
            self.cb_ch.bind("<<ComboboxSelected>>", lambda e: self._show_fc())

        def insert_channel_freq(self):
            b = BLOCKS[self.cb_block.current()]
            ch = b["channels"][self.cb_ch.current()]
            var = self.last_freq_var or (self.gen_rows[0][2][0] if self.gen_rows else None)
            if var is not None:
                var.set("%.6g" % (ch["fc"] / 1e6))
                self._set_status("ch: центр К%03d = %.6g МГц підставлено" % (ch["global"], ch["fc"] / 1e6))

        def _sdr(self):
            return SDR_PRESETS.get(self.cb_sdr.get())

        def _min_width_us(self):
            sp = self._sdr()
            return None if not sp else 4.0 / sp["fs_hz"] * 1e6

        def _show_sdr(self):
            sp = self._sdr()
            if not sp:
                self.lbl_sdr.config(text="пресет не обрано — параметри вручну")
                return
            self.lbl_sdr.config(text="fs ≤ %.6g МС/с, смуга ≤ %.6g МГц, %.6g–%.6g ГГц, %d біт"
                                % (sp["fs_hz"] / 1e6, sp["bw_hz"] / 1e6,
                                   sp["f_lo"] / 1e9, sp["f_hi"] / 1e9, sp["bits"]))

        def _on_sdr_selected(self):
            self._show_sdr()
            sp = self._sdr()
            if sp:
                self.gen_fs.set("%.6g" % (sp["fs_hz"] / 1e6))
                self._set_status("SDR %s: fs ← %.6g МГц, мін. ширина імпульсу ≈ %.4g мкс (4 відліки)"
                                 % (self.cb_sdr.get(), sp["fs_hz"] / 1e6, self._min_width_us()))

        def insert_sdr_width(self):
            var = getattr(self, "last_width_var", None)
            mw = self._min_width_us()
            if var is not None and mw:
                var.set("%.6g" % mw)
                self._set_status("мін. ширину %.6g мкс (4 відліки при %s) підставлено"
                                 % (mw, self.cb_sdr.get()))

        def _current_achx(self):
            if not self.achx_on.get():
                return None
            preset = self.cb_achx.get()
            if preset == "Користувацький":
                pts = parse_achx_table(self.txt_achx.get("1.0", "end"))
            else:
                pts = build_achx_points(preset, parse_freq_hz(self.achx_notch.get()))
            return {"points": pts, "mode": self.achx_mode_key(),
                    "extrap": self.achx_extrap_key()}

        def generate(self):
            try:
                fs_g = parse_freq_hz(self.gen_fs.get()); dur = float(self.gen_dur.get())
                seed = int(self.gen_seed.get())
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні fs/тривалість/seed"); return
            try:
                achx = self._current_achx()
            except ValueError as e:
                messagebox.showerror(APP_TITLE, "АЧХ: %s" % e); return
            params = []
            for en, kind, vals in self.gen_rows:
                try:
                    freq = parse_freq_hz(vals[0].get())
                    phase, amp, width, dc, gain = (float(v.get()) for v in vals[1:])
                except ValueError:
                    messagebox.showerror(APP_TITLE, "Невірне число або частота у параметрах канала"); return
                params.append({"enabled": bool(en.get()), "kind": kind.get(), "freq": freq,
                               "phase": phase, "amp": amp, "width_us": width,
                               "dc": dc, "gain": gain})
            for i, p in enumerate(params):
                if p["enabled"] and p["freq"] > 0 and not (SIG_MIN_HZ <= p["freq"] <= SIG_MAX_HZ):
                    fv, fu = human_freq(p["freq"])
                    messagebox.showerror(
                        APP_TITLE, "ch%d: частота %.4g %s поза частотним планом "
                                   "100 МГц … 6000 МГц (6 блоків)" % (i, fv, fu))
                    return
            mw = self._min_width_us()
            if mw:
                for i, p in enumerate(params):
                    if (p["enabled"] and p["kind"] in ("Меандр", "Імпульс")
                            and 0 < p["width_us"] < mw):
                        if not messagebox.askyesno(
                                APP_TITLE,
                                "ch%d: ширина %.6g мкс менша за мінімальну для %s "
                                "(%.6g мкс = 4 відліки при %.6g МС/с).\n"
                                "Імпульс може бути відтворений некоректно. Генерувати?"
                                % (i, p["width_us"], self.cb_sdr.get(), mw,
                                   self._sdr()["fs_hz"] / 1e6)):
                            return
            n = int(dur * fs_g)
            nch_g = len(params)
            size = n * nch_g * 2
            if size > 8e9:
                messagebox.showerror(APP_TITLE, "Занадто великий файл: %.1f ГБ. Зменшіть тривалість."
                                     % (size / 1e9))
                return
            if size > 500e6 and not messagebox.askyesno(
                    APP_TITLE, "Розмір файла %.0f МБ — продовжити?" % (size / 1e6)):
                return
            fmax = max((p["freq"] for p in params if p["enabled"]), default=0.0)
            if fmax > fs_g / 2:
                fv, fu = human_freq(fmax)
                fn, fun = human_freq(fs_g / 2)
                if not messagebox.askyesno(
                        APP_TITLE,
                        "Максимальна частота %.4g %s вище Найквіста (%.4g %s).\n"
                        "Для прямої оцифровки потрібна fs ≥ %.4g Гц.\n\n"
                        "Генерувати з аліасингом?" % (fv, fu, fn, fun, 2 * fmax)):
                    return
            try:
                extra = {"per_us": float(self.ex_per.get()),
                         "pw_us": float(self.ex_pw.get()),
                         "rise_us": float(self.ex_rise.get()),
                         "fall_us": float(self.ex_fall.get()),
                         "am_m": float(self.ex_am_m.get()),
                         "am_f": float(self.ex_am_f.get()),
                         "chirp_dev": parse_freq_hz(self.ex_chirp.get())}
            except ValueError:
                messagebox.showerror(APP_TITLE, "Невірні додаткові параметри сигналу"); return
            base = os.path.dirname(self.path) if getattr(self, "path", None) else os.getcwd()
            out = filedialog.asksaveasfilename(initialdir=base, defaultextension=".raw",
                                               initialfile="generated_%dch_%dHz.raw" % (nch_g, int(fs_g)),
                                               title="Куди зберегти новий .raw",
                                               filetypes=[("RAW файли", "*.raw")])
            if not out:
                return
            self._set_status("генерую %d×%d відліків%s…" % (n, nch_g, ", з АЧХ" if achx else ""))
            flat = generate_raw(params, n, fs_g, nch_g, seed, achx=achx, extra=extra)
            save_raw(out, flat)
            self._set_status("записано %s (%.1f МБ) — відкриваю…" % (os.path.basename(out), os.path.getsize(out) / 1e6))
            self.fs_var.set(self.gen_fs.get()); self.nch_var.set(str(nch_g))
            self._load(out)

    app = App()
    app.mainloop()


# ----------------------------------------------------------------------------
# CLI-режими
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument("--headless", nargs="+", metavar=("FILE", "fs_nch"))
    ap.add_argument("--selftest", nargs="*", metavar="FILE")
    ap.add_argument("--preview", nargs="+", metavar=("OUT_PREFIX", "FILE"))
    ap.add_argument("--calgen", nargs="+", metavar=("OUT", "fs_nch_tones_fmin_fmax_amp_seg"))
    ap.add_argument("--calmeas", nargs="+", metavar=("RECORDED", "PLAN"))
    args = ap.parse_args()

    if args.selftest is not None:
        return selftest(args.selftest[0] if args.selftest else "P-2-2.raw")

    if args.headless is not None:
        p = args.headless[0]
        fsv = float(args.headless[1]) if len(args.headless) > 1 else DEFAULT_FS
        nch = int(args.headless[2]) if len(args.headless) > 2 else DEFAULT_NCH
        rep, _, _ = analyze_report(p, fsv, nch)
        print(rep)
        return 0

    if args.preview is not None and len(args.preview) >= 2:
        prefix, p = args.preview[0], args.preview[1]
        fsv = float(args.preview[2]) if len(args.preview) > 2 else DEFAULT_FS
        nch = int(args.preview[3]) if len(args.preview) > 3 else DEFAULT_NCH
        return preview(prefix, p, fsv, nch)

    if args.calgen is not None and len(args.calgen) >= 3:
        out = args.calgen[0]
        fs_c, nch_c = float(args.calgen[1]), int(args.calgen[2])
        tones = int(args.calgen[3]) if len(args.calgen) > 3 else 32
        fmin = float(args.calgen[4]) if len(args.calgen) > 4 else SIG_MIN_HZ
        fmax = float(args.calgen[5]) if len(args.calgen) > 5 else SIG_MAX_HZ
        amp = float(args.calgen[6]) if len(args.calgen) > 6 else 8000.0
        seg_n = int(args.calgen[7]) if len(args.calgen) > 7 else 16384
        freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
        save_raw(out, generate_cal_file(fs_c, nch_c, freqs, amp, seg_n))
        save_cal_plan(out + ".plan", fs_c, nch_c, freqs, amp, seg_n)
        print("файл: %s (%.1f МБ), план: %s (%d тонів, бін %.3f Гц)"
              % (out, os.path.getsize(out) / 1e6, out + ".plan", len(freqs), bin_hz))
        print("\n".join("  %.6g Гц" % fq for fq in freqs))
        return 0

    if args.calmeas is not None and len(args.calmeas) >= 2:
        plan = load_cal_plan(args.calmeas[1])
        res = measure_achx(args.calmeas[0], plan)
        print("частота_Гц  ампл_кодів  дБ_відн_еталону(%d)" % plan["amp"])
        for fq, amp, gdb in res:
            print("%12.6g  %10.1f  %+6.2f" % (fq, amp, gdb))
        return 0

    try:
        run_gui()
    except ImportError as e:
        print("GUI недоступний (%s). Використайте: --headless / --calgen / --calmeas" % e)
        return 1
    return 0


def preview(prefix, path, fs, nch):
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["font.family"] = "DejaVu Sans"
    from matplotlib.figure import Figure
    blocks = build_blocks()
    chans, n = load_raw(path, nch)
    fig = Figure(figsize=(15, 5.5), dpi=110)
    draw_time(fig, chans, fs, 0.0, min(1e-5, n / fs))
    fig.savefig(prefix + "_time.png", bbox_inches="tight")
    rows, cols = _grid_for(nch)
    fig2 = Figure(figsize=(15, 3.2 * rows + 1), dpi=110)
    peaks = draw_spectrum(fig2, chans, fs)
    fig2.savefig(prefix + "_spectrum.png", bbox_inches="tight")
    fig3 = Figure(figsize=(14, 12), dpi=110)
    occ = channel_occupancy(chans, fs, blocks)
    info = draw_blocks(fig3, chans, fs, blocks, occ)
    fig3.savefig(prefix + "_blocks.png", bbox_inches="tight")
    print("\n".join(peaks)); print("\n".join(info))
    print("saved: %s_time.png, %s_spectrum.png, %s_blocks.png" % (prefix, prefix, prefix))
    return 0


def selftest(path):
    ok = True
    print("--- ПЕРЕВІРКА ЧАСТОТНОГО ПЛАНУ (6 блоків, 64 МГц) ---")
    blocks = build_blocks()
    full = [sum(1 for c in b["channels"] if not c["partial"]) for b in blocks]
    part = [sum(1 for c in b["channels"] if c["partial"]) for b in blocks]
    total = sum(len(b["channels"]) for b in blocks)
    last = blocks[-1]["channels"][-1]
    print("повних по блоках:", full, "(очік. [6,18,20,18,18,25], сума 105)")
    print("крайових:", part, "(очік. [1,1,1,1,1,0])")
    ok &= full == [6, 18, 20, 18, 18, 25] and part == [1, 1, 1, 1, 1, 0]
    ok &= total == 110 and abs(last["f1"] - 6000e6) < 1.0

    if os.path.exists(path):
        rep, _, _ = analyze_report(path, 1_000_000, 4)
        print("--- аналіз реального файла (4 канали, fs=1 МГц) ---")
        print("\n".join(rep.splitlines()[:10]))

    print("--- RF-ТЕСТ: fs = 12 ГГц, 8 каналів ---")
    fsrf, nchrf, nrf = 12e9, 8, 1_200_000
    params_rf = [
        {"enabled": True, "kind": "Синус", "freq": 100e6, "phase": 0, "amp": 12000, "width_us": 2, "dc": 32768, "gain": 1.0},
        {"enabled": True, "kind": "Синус", "freq": 852e6, "phase": 45, "amp": 12000, "width_us": 2, "dc": 30000, "gain": 1.0},
        {"enabled": True, "kind": "Шум", "freq": 0, "phase": 0, "amp": 700, "width_us": 0, "dc": 40000, "gain": 1.0},
        {"enabled": True, "kind": "Імпульс", "freq": 60e6, "phase": 0, "amp": 25000, "width_us": 0.002, "dc": 5000, "gain": 1.0},
    ] + [{"enabled": False, "kind": "Синус", "freq": 0, "phase": 0, "amp": 0,
          "width_us": 0, "dc": 1000 * (k + 5), "gain": 1.0} for k in range(4)]
    tmp_rf = "/tmp/opencode/rawstudio_rf8.raw"
    save_raw(tmp_rf, generate_raw(params_rf, nrf, fsrf, nchrf, seed=13))
    rrf, _ = load_raw(tmp_rf, nchrf)
    fr, ar = amplitude_spectrum(rrf[0], fsrf)
    pk0 = find_peaks(fr, ar, k=1, exclude_bins=5)[0]
    print("ch0: пік = %.4g Гц, ампл. %.0f (очік. 100 МГц / 12000)" % (pk0[0], pk0[1]))
    ok &= abs(pk0[0] - 100e6) < 1e6 and abs(pk0[1] - 12_000) < 2_500
    occ = channel_occupancy(rrf, fsrf, blocks)
    target = [c for c in blocks[1]["channels"] if abs(c["fc"] - 852e6) < 1e3][0]
    print("канал 852 МГц: К%03d, зайнятий: %s" % (target["global"], occ[target["global"]]["occupied"]))
    ok &= occ[target["global"]]["occupied"]

    # --- АЧХ-модуль: тони точно в бінах ---
    print("--- ТЕСТ АЧХ: амплітуда залежно від частоти ---")
    n_a = 1_200_000
    t = np.arange(n_a) / fsrf
    w = 10000 * np.sin(2 * np.pi * 100e6 * t) + 10000 * np.sin(2 * np.pi * 3e9 * t)
    pts = parse_achx_table("100e6 0\n3e9 -6.0206\n6e9 -12.0412")
    w_f = apply_achx(w, fsrf, pts)
    f_a, a_f = amplitude_spectrum(w_f, fsrf, nfft=n_a)
    a100 = float(a_f[np.argmin(np.abs(f_a - 100e6))])
    a3g = float(a_f[np.argmin(np.abs(f_a - 3e9))])
    print("з АЧХ: A(100М)=%.0f (очік. 10000), A(3Г)=%.0f (очік. 5000)" % (a100, a3g))
    ok &= abs(a100 - 10_000) < 300 and abs(a3g - 5_000) < 300

    # --- КАЛІБРУВАННЯ: замкнений контур ---
    print("--- ТЕСТ КАЛІБРУВАННЯ: прогін тонів → штучний тракт → вимірювання АЧХ ---")
    fs_c, nch_c = 12e9, 8
    tones, fmin, fmax, amp_ref, seg_n = 12, 100e6, 6000e6, 8000.0, 16384
    freqs, bin_hz = make_cal_plan(fs_c, fmin, fmax, tones, seg_n)
    tract = parse_achx_table("100e6 0\n1e9 -3\n3e9 -6\n6e9 -12")
    cal = "/tmp/opencode/cal_src.raw"
    rec = "/tmp/opencode/cal_after_tract.raw"
    plan_p = cal + ".plan"
    save_raw(cal, generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n))
    save_raw(rec, generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n, tract_points=tract))
    save_cal_plan(plan_p, fs_c, nch_c, freqs, amp_ref, seg_n)
    res_src = measure_achx(cal, load_cal_plan(plan_p))
    res_rec = measure_achx(rec, load_cal_plan(plan_p))
    ctrl = max(abs(m[2]) for m in res_src)
    err = max(abs(m[2] - float(np.interp(m[0], [p[0] for p in tract], [p[1] for p in tract])))
              for m in res_rec)
    print("контроль без тракту: %.3f дБ; вимірювання тракту: макс. похибка %.3f дБ (очік. < 0.6)"
          % (ctrl, err))
    ok &= ctrl < 0.3 and err < 0.6
    combined = [(m[0], float(np.interp(m[0], [p[0] for p in tract], [p[1] for p in tract])) - m[2])
                for m in res_rec]
    save_raw("/tmp/opencode/cal_flat.raw",
             generate_cal_file(fs_c, nch_c, freqs, amp_ref, seg_n, tract_points=combined))
    flat_max = max(abs(m[2]) for m in measure_achx("/tmp/opencode/cal_flat.raw",
                                                   load_cal_plan(plan_p)))
    print("після компенсації: макс. відхилення від 0 дБ = %.3f дБ (очік. < 1.5)" % flat_max)
    ok &= flat_max < 1.5

    # --- ІНТЕРПОЛЯЦІЯ АЧХ: сплайн/log/екстраполяція ---
    print("--- ТЕСТ ІНТЕРПОЛЯЦІЇ АЧХ: будь-яка частота між точками і за межами ---")
    pts_i = [(1e8, 0.0), (3e9, -6.0), (6e9, -12.0)]
    knots = np.array([1e8, 3e9, 6e9])
    want = 10 ** (np.array([0.0, -6.0, -12.0]) / 20.0)
    h_sp = achx_response(knots, pts_i, mode="spline")
    h_lg = achx_response(knots, pts_i, mode="logspline")
    print("сплайн через вузли: %s" % np.allclose(h_sp, want, rtol=1e-9))
    print("log-сплайн через вузли: %s" % np.allclose(h_lg, want, rtol=1e-9))
    ok &= np.allclose(h_sp, want, rtol=1e-9) and np.allclose(h_lg, want, rtol=1e-9)
    mid_sp = float(achx_response(np.array([1.5e9]), pts_i, mode="spline")[0])
    mid_ln = float(achx_response(np.array([1.5e9]), pts_i, mode="linear")[0])
    print("у проміжку 1.5 ГГц: сплайн=%.4f, лінійна=%.4f (різні методи — обидва валідні)" % (mid_sp, mid_ln))
    ok &= 0.3 < mid_sp < 0.8 and 0.3 < mid_ln < 0.8
    h_cl = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="clamp")[0])
    h_li = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="linear")[0])
    h_ze = float(achx_response(np.array([8e9]), pts_i, mode="spline", extrap="zero")[0])
    print("поза діапазоном 8 ГГц: clamp=%.4f (очік. 0.2512), linear=%.4f (продовження вниз), zero=%.4f (очік. 0)"
          % (h_cl, h_li, h_ze))
    ok &= abs(h_cl - 0.25119) < 1e-3 and h_li < h_cl and h_ze == 0.0
    w_lg = apply_achx(10000 * np.sin(2 * np.pi * 1.5e9 * t), fsrf, pts_i, mode="logspline")
    print("генерація з log-сплайном на 1.5 ГГц: rms=%.0f кодів" % float(np.std(w_lg)))
    ok &= 3000 < float(np.std(w_lg)) < 6000

    # --- ПРОФІЛЬ У ФАЙЛ / НАЗАД ---
    print("--- ТЕСТ ПРОФІЛЮ АЧХ: збереження/завантаження (.achx, з режимами) ---")
    prof_path = "/tmp/opencode/achx_measured.achx"
    save_achx_profile(prof_path, [(m[0], m[2]) for m in res_rec],
                      mode="spline", extrap="linear", meta={"fs": "%g" % fs_c})
    pts_back, mode_b, ex_b = load_achx_profile(prof_path)
    df = max(abs(a[0] / b[0] - 1.0) for a, b in zip(pts_back, res_rec))
    dg = max(abs(a[1] - b[2]) for a, b in zip(pts_back, res_rec))
    print("точок: %d, похибка частот: %.2g, дБ: %.2g, режими: %s/%s (очік. spline/linear)"
          % (len(pts_back), df, dg, mode_b, ex_b))
    ok &= len(pts_back) == len(res_rec) and df < 1e-5 and dg < 1e-3
    ok &= mode_b == "spline" and ex_b == "linear"

    print("ПІДСУМОК:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
