"""
ECG Image Cleaner — Fourier wrinkle removal, RED GRID PRESERVED
================================================================
Removes wrinkles / shadow shading from a scanned ECG while keeping
the original red/pink grid lines intact in colour.

Strategy:
  1. Extract the red-grid mask from the colour image
  2. Build a "grid-free" greyscale (traces + paper only)
  3. FFT: notch-filter residual grid periodicity
  4. FFT: low-freq background separation → flatten wrinkle shading
  5. Contrast enhancement on the cleaned greyscale  ← ADAPTIVE (fixed)
  6. Bolden ECG traces via morphological dilation
  7. Re-composite: clean white paper + red grid overlay + bold dark traces

Usage:
    python ecg_clean_keep_grid.py input.png output.png
    python ecg_clean_keep_grid.py input.png output.png --bg-cutoff 60 --debug
"""

import numpy as np
from PIL import Image
import argparse, os


def to_float(a):  return a.astype(np.float64) / 255.0
def to_uint8(a):  return np.clip(a * 255, 0, 255).astype(np.uint8)


# ── 1. Extract red grid mask ────────────────────────────────────

def extract_red_grid(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    redness = np.clip(r - 0.5 * (g + b), 0, 1)
    brightness = (r + g + b) / 3.0
    raw_mask = (redness * 3.5) * np.clip(brightness * 2 - 0.2, 0, 1)
    grid_mask = np.clip(raw_mask, 0, 1)

    from PIL import ImageFilter
    gm_img = Image.fromarray(to_uint8(grid_mask), "L")
    gm_img = gm_img.filter(ImageFilter.GaussianBlur(radius=1))
    grid_mask = to_float(np.array(gm_img))

    grid_colour = rgb.copy()
    grey = 0.2989 * r + 0.5870 * g + 0.1140 * b
    grey_clean = grey * (1 - grid_mask) + 1.0 * grid_mask

    return grid_mask, grid_colour, grey_clean


# ── 2. FFT grid-peak detection & notch filtering ────────────────

def find_grid_peaks(grey, threshold_factor=4.0,
                    min_spacing=12, max_spacing=120):
    rows, cols = grey.shape
    F = np.fft.fftshift(np.fft.fft2(grey))
    mag = np.abs(F)
    cy, cx = rows // 2, cols // 2

    def _fundamental(profile, N):
        half = profile[N // 2:]
        half[:5] = 0
        lo = max(1, N // max_spacing)
        hi = min(len(half) - 1, N // min_spacing)
        if lo >= hi: return None
        roi = half[lo:hi]
        thr = np.median(roi) + threshold_factor * np.std(roi)
        cands = np.where(roi > thr)[0] + lo
        return cands[np.argmax(half[cands])] if len(cands) else None

    ky = _fundamental(mag.mean(axis=1), rows)
    kx = _fundamental(mag.mean(axis=0), cols)
    if ky: print(f"  Horiz. grid: ~{rows/ky:.1f} px  (k={ky})")
    if kx: print(f"  Vert.  grid: ~{cols/kx:.1f} px  (k={kx})")

    peaks = set()
    nh = (rows // 2) // ky if ky else 0
    nw = (cols // 2) // kx if kx else 0
    for ny in range(-nh, nh + 1):
        for nx in range(-nw, nw + 1):
            if ny == 0 and nx == 0: continue
            fy = cy + ny * (ky or 0)
            fx = cx + nx * (kx or 0)
            if 0 <= fy < rows and 0 <= fx < cols:
                peaks.add((int(fy), int(fx)))
    return list(peaks)


def apply_notch_filter(grey, peaks, radius=5):
    rows, cols = grey.shape
    mask = np.ones((rows, cols), dtype=np.float64)
    Y, X = np.ogrid[:rows, :cols]
    for py, px in peaks:
        d2 = (Y - py)**2 + (X - px)**2
        mask *= (1 - np.exp(-d2 / (2 * radius**2)))
    cy, cx = rows // 2, cols // 2
    mask[cy, cx] = 1.0
    F = np.fft.fftshift(np.fft.fft2(grey))
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
    return np.clip(result, 0, 1)


# ── 3. Wrinkle flattening ───────────────────────────────────────

def flatten_background(grey, cutoff=60):
    rows, cols = grey.shape
    cy, cx = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    D2 = (Y - cy)**2 + (X - cx)**2
    lp = np.exp(-D2 / (2 * cutoff**2))
    F = np.fft.fftshift(np.fft.fft2(grey))
    bg = np.real(np.fft.ifft2(np.fft.ifftshift(F * lp)))
    bg = np.clip(bg, 0.01, None)
    flat = grey / bg
    return flat / flat.max(), bg


# ── 4. Contrast — adaptive white-point only ────────────────────

def enhance_contrast(grey, gamma=0.40):
    """
    Two-ended adaptive stretch — consistent output on both foggy and normal scans.

    Why a black-point stretch is now safe:
      The flatten_background step (bg_cutoff=120) divides out slow-varying
      illumination gradients — including crease/wrinkle shadows — before we
      get here.  After that division the only remaining dark pixels are ECG
      traces, so anchoring the black end to the 1st percentile stretches
      traces to black without amplifying shadows.
      (With the old bg_cutoff=60 the shadow was NOT removed, causing the
      crease-line artefact.  bg_cutoff=120 fixes the root cause.)

    Safety cap: black-point is capped at 0.50 so a pathological image with
    unusually deep residual shadow can never produce a more than 2× stretch.
    """
    # white-point: paper → 1.0
    wp = np.percentile(grey, 99)
    grey = np.clip(grey / (wp + 1e-6), 0, 1)

    # gamma brightens background, barely moves already-dark traces
    grey = grey ** gamma

    # black-point: anchor darkest trace pixels → 0
    # cap at 0.50 as a safety rail against any residual shadow
    bp = min(float(np.percentile(grey, 1)), 0.50)
    grey = np.clip((grey - bp) / (1.0 - bp + 1e-6), 0, 1)

    return grey


# ── 5. Bolden ECG traces ────────────────────────────────────────

def bolden_traces(grey, dark_threshold=0.7, dilate_size=3, blend=0.6):
    from PIL import ImageFilter

    darkness = np.clip((dark_threshold - grey) / dark_threshold, 0, 1)
    darken_amount = darkness ** 0.5
    darkened = grey * (1 - darken_amount * 0.7)
    darkened = np.clip(darkened, 0, 1)

    dark_pil = Image.fromarray(to_uint8(darkened), "L")
    dilated_pil = dark_pil.filter(ImageFilter.MinFilter(size=dilate_size))
    dilated = to_float(np.array(dilated_pil))

    trace_mask = (darkened < dark_threshold).astype(np.float64)
    mask_pil = Image.fromarray(to_uint8(trace_mask), "L")
    mask_pil = mask_pil.filter(ImageFilter.MaxFilter(size=dilate_size + 2))
    trace_mask = to_float(np.array(mask_pil))

    result = darkened * (1 - trace_mask * blend) + dilated * (trace_mask * blend)
    return np.clip(result, 0, 1)


# ── 6. Re-composite with grid ───────────────────────────────────

def composite_with_grid(clean_grey, grid_mask, grid_colour,
                        grid_opacity=0.55, grid_saturation=1.4):
    h, w = clean_grey.shape
    out = np.stack([clean_grey] * 3, axis=-1)

    gr, gg, gb = grid_colour[:,:,0], grid_colour[:,:,1], grid_colour[:,:,2]
    avg = (gr + gg + gb) / 3.0
    gc = grid_colour.copy()
    for c, ch in enumerate([gr, gg, gb]):
        gc[:,:,c] = np.clip(avg + (ch - avg) * grid_saturation, 0, 1)

    alpha = (grid_mask * grid_opacity)[:, :, np.newaxis]
    out = out * (1 - alpha) + gc * alpha

    return np.clip(out, 0, 1)


# ── Pipeline ─────────────────────────────────────────────────────

def clean_ecg(input_path, output_path,
              notch_radius=5,
              bg_cutoff=120,
              gamma=0.40,
              bold_threshold=0.75,
              bold_size=3,
              bold_blend=0.8,
              grid_opacity=0.55,
              grid_saturation=1.4,
              debug=False):

    img = Image.open(input_path).convert("RGB")
    rgb = to_float(np.array(img))
    h, w = rgb.shape[:2]
    print(f"Image: {w}x{h}")

    print("1. Extracting red grid mask...")
    grid_mask, grid_colour, grey = extract_red_grid(rgb)

    if debug:
        _save_debug(output_path, "1a_grid_mask", grid_mask)
        _save_debug(output_path, "1b_grey_no_grid", grey)

    print("2. FFT notch filter (residual grid harmonics)...")
    peaks = find_grid_peaks(grey)
    print(f"   {len(peaks)} peaks")
    if peaks:
        grey = apply_notch_filter(grey, peaks, notch_radius)

    if debug:
        _save_debug(output_path, "2_after_notch", grey)

    print("3. FFT background flattening...")
    grey, bg = flatten_background(grey, bg_cutoff)

    if debug:
        _save_debug(output_path, "3a_background", bg / bg.max())
        _save_debug(output_path, "3b_flattened", grey)

    print("4. Adaptive contrast enhancement...")
    grey = enhance_contrast(grey, gamma)          # white_pct removed — now adaptive

    if debug:
        _save_debug(output_path, "4_contrast", grey)

    print("5. Boldening ECG traces...")
    grey = bolden_traces(grey, dark_threshold=bold_threshold,
                         dilate_size=bold_size, blend=bold_blend)

    if debug:
        _save_debug(output_path, "5_bolded", grey)

    print("6. Compositing red grid back...")
    result = composite_with_grid(grey, grid_mask, grid_colour,
                                 grid_opacity, grid_saturation)

    Image.fromarray(to_uint8(result), "RGB").save(output_path)
    print(f"Saved: {output_path}")


def _save_debug(output_path, tag, img):
    base, ext = os.path.splitext(output_path)
    if img.ndim == 2:
        Image.fromarray(to_uint8(img), "L").save(f"{base}_{tag}.png")
    else:
        Image.fromarray(to_uint8(img), "RGB").save(f"{base}_{tag}.png")


# ── CLI ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import glob as _glob

    pa = argparse.ArgumentParser(
        description="Clean scanned ECG (Fourier), preserving red grid.")
    pa.add_argument("input",  help="Input image or directory of PNGs")
    pa.add_argument("output", help="Output image or directory")
    pa.add_argument("--notch-radius", type=int, default=5)
    pa.add_argument("--bg-cutoff", type=float, default=120)
    pa.add_argument("--gamma", type=float, default=0.40)
    pa.add_argument("--bold-threshold", type=float, default=0.75)
    pa.add_argument("--bold-size", type=int, default=3)
    pa.add_argument("--bold-blend", type=float, default=0.8)
    pa.add_argument("--grid-opacity", type=float, default=0.55)
    pa.add_argument("--grid-saturation", type=float, default=1.4)
    pa.add_argument("--debug", action="store_true")
    a = pa.parse_args()

    kwargs = dict(
        notch_radius=a.notch_radius,
        bg_cutoff=a.bg_cutoff,
        gamma=a.gamma,
        bold_threshold=a.bold_threshold,
        bold_size=a.bold_size,
        bold_blend=a.bold_blend,
        grid_opacity=a.grid_opacity,
        grid_saturation=a.grid_saturation,
        debug=a.debug,
    )

    if os.path.isdir(a.input):
        os.makedirs(a.output, exist_ok=True)
        pngs = sorted(_glob.glob(os.path.join(a.input, "*.png")))
        total = len(pngs)
        skipped = 0
        for i, src in enumerate(pngs):
            fname = os.path.basename(src)
            dst = os.path.join(a.output, fname)
            if os.path.exists(dst):
                skipped += 1
                continue
            print(f"[{i+1}/{total}] {fname}")
            clean_ecg(src, dst, **kwargs)
        print(f"Done. {total - skipped} processed, {skipped} skipped.")
    else:
        clean_ecg(a.input, a.output, **kwargs)