# Accentuate white vs gray matter via window/level LUTs
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---- load image (grayscale) ----
img_path = r"a1images\brain_proton_density_slice.png"
f = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
if f is None:
    raise FileNotFoundError(f"Could not read image at: {img_path}")

# ---- helper: build window/level LUT (piecewise-linear) ----
def window_lut(level: float, width: float, out_low=0, out_high=255) -> np.ndarray:
    """Return uint8 LUT t[0..255] that linearly maps [level-width/2, level+width/2] to [out_low,out_high]."""
    x = np.arange(256, dtype=np.float32)
    lo, hi = level - width/2.0, level + width/2.0
    y = np.empty_like(x)
    # below window
    y[x <= lo] = out_low
    # above window
    y[x >= hi] = out_high
    # inside window: linear ramp
    m = (x > lo) & (x < hi)
    y[m] = (x[m] - lo) / (hi - lo) * (out_high - out_low) + out_low
    return np.clip(y, 0, 255).astype(np.uint8)

# ---- choose windows (tune after viewing histogram) ----
# For PD: WM darker than GM
WM_LEVEL, WM_WIDTH = 100, 70   # emphasize white matter band
GM_LEVEL, GM_WIDTH = 145, 70   # emphasize gray matter band

t_wm = window_lut(WM_LEVEL, WM_WIDTH)   # LUT to accentuate WM
t_gm = window_lut(GM_LEVEL, GM_WIDTH)   # LUT to accentuate GM

# ---- apply LUTs ----
g_wm = t_wm[f]
g_gm = t_gm[f]

# ---- visualize: images, hist, and the transformation functions ----
fig, ax = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)

# original + histogram
ax[0,0].imshow(f, cmap='gray', vmin=0, vmax=255); ax[0,0].set_title("Original"); ax[0,0].axis('off')
ax[1,0].hist(f.ravel(), bins=256, range=(0,255)); ax[1,0].set_title("Histogram"); ax[1,0].set_xlim(0,255)

# WM accentuated
ax[0,1].imshow(g_wm, cmap='gray', vmin=0, vmax=255); ax[0,1].set_title("Accentuate WHITE matter"); ax[0,1].axis('off')
ax[1,1].plot(np.arange(256), t_wm); ax[1,1].set_xlim(0,255); ax[1,1].set_ylim(0,255); ax[1,1].set_title("WM Transform")

# GM accentuated
ax[0,2].imshow(g_gm, cmap='gray', vmin=0, vmax=255); ax[0,2].set_title("Accentuate GRAY matter"); ax[0,2].axis('off')
ax[1,2].plot(np.arange(256), t_gm); ax[1,2].set_xlim(0,255); ax[1,2].set_ylim(0,255); ax[1,2].set_title("GM Transform")

plt.show()
