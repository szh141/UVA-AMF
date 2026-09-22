"""
Frequency Domain Explorer
==========================

An interactive matplotlib app for building intuition about the 2D Fourier
transform of an image: what the DC term is, what "low frequency" and "high
frequency" mean spatially, and what happens to an image when you keep or
remove different bands of frequencies.

Layout
------
  [ Original image ]   [ Log-magnitude spectrum + filter mask overlay ]
  [ Reconstructed  ]   [ Filtered spectrum (what actually got kept)   ]

Controls (bottom panel)
------------------------
  - Radio buttons : filter type
        None        - identity, no filtering
        Low-pass    - keep a disk of radius r1 around the DC term (center)
        High-pass   - remove a disk of radius r1 around the DC term
        Band-pass   - keep only an annulus between r1 and r2
        Band-stop   - remove only an annulus between r1 and r2
  - Slider r1       : inner/primary cutoff radius, in cycles-per-image units
  - Slider r2       : outer cutoff radius (used only for band-pass/band-stop)
  - Check "Zero DC" : additionally zero out the single DC (mean-brightness)
                      coefficient, to show what removing "zero frequency"
                      alone does to an image (it removes the mean/average
                      brightness -- the image becomes a zero-mean edge map).
  - Reset button    : restore defaults

Run
---
    python freq_domain_explorer.py                  # uses a built-in sample image
    python freq_domain_explorer.py path/to/image.png # uses your own image

Requires: numpy, scipy, scikit-image, matplotlib (with an interactive
backend such as TkAgg or Qt5Agg -- the default backend on most desktop
Python installs is already interactive).
"""

import sys

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

from skimage import data, img_as_float
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, CheckButtons, Button


# --------------------------------------------------------------------------
# 1. Load the image
# --------------------------------------------------------------------------
def load_image(path=None, max_size=512):
    """Load a grayscale float image, either from disk or a built-in sample."""
    if path is not None:
        img = imread(path)
        if img.ndim == 3:
            img = rgb2gray(img[..., :3])
        img = img_as_float(img)
    else:
        # skimage.data.camera() is a classic image-processing test image --
        # sharp edges (high frequency) and smooth shading (low frequency)
        # side by side, which makes it a good teaching image.
        img = img_as_float(data.camera())

    # Keep things responsive: downsize very large images.
    h, w = img.shape
    scale = max_size / max(h, w)
    if scale < 1.0:
        img = resize(img, (int(h * scale), int(w * scale)), anti_aliasing=True)
    return img


# --------------------------------------------------------------------------
# 2. Frequency-domain machinery
# --------------------------------------------------------------------------
def radial_grid(shape):
    """Return a grid of radial distance (in pixels) from the DC term (center)
    once the spectrum has been fftshift-ed, plus the max radius for scaling
    sliders."""
    h, w = shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_max = np.sqrt(cy ** 2 + cx ** 2)
    return r, r_max


def build_mask(shape, filter_type, r1, r2, zero_dc):
    """Build a binary frequency-domain mask (applied to the *shifted*
    spectrum, so index [center] = DC)."""
    r, _ = radial_grid(shape)
    h, w = shape
    cy, cx = h // 2, w // 2

    if filter_type == "None":
        mask = np.ones(shape, dtype=float)
    elif filter_type == "Low-pass":
        mask = (r <= r1).astype(float)
    elif filter_type == "High-pass":
        mask = (r > r1).astype(float)
    elif filter_type == "Band-pass":
        lo, hi = min(r1, r2), max(r1, r2)
        mask = ((r >= lo) & (r <= hi)).astype(float)
    elif filter_type == "Band-stop":
        lo, hi = min(r1, r2), max(r1, r2)
        mask = ((r < lo) | (r > hi)).astype(float)
    else:
        mask = np.ones(shape, dtype=float)

    if zero_dc:
        mask[cy, cx] = 0.0

    return mask


def apply_filter(img, filter_type, r1, r2, zero_dc):
    """FFT -> shift -> mask -> unshift -> inverse FFT. Returns the
    reconstructed (real) image, the shifted log-magnitude spectrum, the
    mask used, and the shifted log-magnitude spectrum *after* filtering."""
    F = fft2(img)
    Fshift = fftshift(F)

    mask = build_mask(img.shape, filter_type, r1, r2, zero_dc)
    Fshift_filtered = Fshift * mask

    # Back to an image
    F_filtered = ifftshift(Fshift_filtered)
    reconstructed = np.real(ifft2(F_filtered))

    mag_full = np.log1p(np.abs(Fshift))
    mag_filtered = np.log1p(np.abs(Fshift_filtered))

    return reconstructed, mag_full, mask, mag_filtered, Fshift


# --------------------------------------------------------------------------
# 3. The interactive app
# --------------------------------------------------------------------------
def run_app(image_path=None):
    img = load_image(image_path)
    h, w = img.shape
    _, r_max = radial_grid(img.shape)

    # Sensible defaults: a low-pass filter keeping the central ~15% of the
    # frequency radius, which is usually enough to show heavy blurring.
    state = {
        "filter_type": "Low-pass",
        "r1": r_max * 0.15,
        "r2": r_max * 0.35,
        "zero_dc": False,
    }

    fig = plt.figure(figsize=(12, 11))
    fig.suptitle("Frequency Domain Explorer  —  2D FFT of an image", fontsize=13, y=0.995)

    ax_orig = fig.add_axes([0.06, 0.59, 0.40, 0.30])
    ax_spec = fig.add_axes([0.52, 0.59, 0.40, 0.30])
    ax_recon = fig.add_axes([0.06, 0.26, 0.40, 0.30])
    ax_specf = fig.add_axes([0.52, 0.26, 0.40, 0.30])

    for ax in (ax_orig, ax_spec, ax_recon, ax_specf):
        ax.set_xticks([])
        ax.set_yticks([])

    ax_orig.set_title("Original image")
    ax_spec.set_title("Log-magnitude spectrum  (center = DC)")
    ax_recon.set_title("Reconstructed image (after filtering)")
    ax_specf.set_title("Spectrum actually kept (mask applied)")

    im_orig = ax_orig.imshow(img, cmap="gray")

    recon0, mag0, mask0, magf0, _ = apply_filter(img, **state)
    im_spec = ax_spec.imshow(mag0, cmap="magma")
    im_recon = ax_recon.imshow(recon0, cmap="gray", vmin=0, vmax=1)
    im_specf = ax_specf.imshow(magf0, cmap="magma", vmin=mag0.min(), vmax=mag0.max())

    # Overlay circles on the spectrum panel to show the current cutoff radii
    cy, cx = h // 2, w // 2
    circ1 = plt.Circle((cx, cy), state["r1"], fill=False, edgecolor="cyan", linewidth=1.5)
    circ2 = plt.Circle((cx, cy), state["r2"], fill=False, edgecolor="lime",
                        linewidth=1.5, linestyle="--")
    ax_spec.add_patch(circ1)
    ax_spec.add_patch(circ2)
    dc_marker, = ax_spec.plot([cx], [cy], marker="+", color="white", markersize=10,
                              markeredgewidth=2)

    info_text = fig.text(0.06, 0.935, "", fontsize=9, family="monospace")

    # ---- Widgets --------------------------------------------------------
    ax_radio = fig.add_axes([0.06, 0.03, 0.15, 0.19])
    radio = RadioButtons(ax_radio, ("None", "Low-pass", "High-pass",
                                     "Band-pass", "Band-stop"), active=1)

    ax_r1 = fig.add_axes([0.28, 0.18, 0.38, 0.025])
    slider_r1 = Slider(ax_r1, "r1 (cutoff)", 0.0, r_max, valinit=state["r1"])

    ax_r2 = fig.add_axes([0.28, 0.12, 0.38, 0.025])
    slider_r2 = Slider(ax_r2, "r2 (band outer)", 0.0, r_max, valinit=state["r2"])

    ax_check = fig.add_axes([0.28, 0.03, 0.18, 0.06])
    check = CheckButtons(ax_check, ["Zero DC"], [False])

    ax_reset = fig.add_axes([0.50, 0.03, 0.14, 0.05])
    button_reset = Button(ax_reset, "Reset")

    def redraw():
        recon, mag, mask, magf, Fshift = apply_filter(img, **state)

        im_recon.set_data(np.clip(recon, 0, 1))
        im_specf.set_data(magf)
        im_specf.set_clim(mag.min(), mag.max())

        circ1.center = (cx, cy)
        circ1.radius = state["r1"]
        circ2.center = (cx, cy)
        circ2.radius = state["r2"]
        circ2.set_visible(state["filter_type"] in ("Band-pass", "Band-stop"))

        kept_frac = mask.mean() * 100
        dc_val = np.abs(Fshift[cy, cx]) / (h * w)
        mean_recon = recon.mean()
        info_text.set_text(
            f"filter: {state['filter_type']:<10s}  "
            f"kept: {kept_frac:5.1f}% of frequency components   "
            f"|DC| (mean brightness proxy): {dc_val:6.3f}   "
            f"reconstructed mean: {mean_recon:6.3f}"
        )
        fig.canvas.draw_idle()

    def on_filter_change(label):
        state["filter_type"] = label
        redraw()

    def on_r1_change(val):
        state["r1"] = val
        redraw()

    def on_r2_change(val):
        state["r2"] = val
        redraw()

    def on_check(label):
        state["zero_dc"] = not state["zero_dc"]
        redraw()

    def on_reset(event):
        state["filter_type"] = "Low-pass"
        state["r1"] = r_max * 0.15
        state["r2"] = r_max * 0.35
        state["zero_dc"] = False
        radio.set_active(1)
        slider_r1.set_val(state["r1"])
        slider_r2.set_val(state["r2"])
        if check.get_status()[0]:
            check.set_active(0)
        redraw()

    radio.on_clicked(on_filter_change)
    slider_r1.on_changed(on_r1_change)
    slider_r2.on_changed(on_r2_change)
    check.on_clicked(on_check)
    button_reset.on_clicked(on_reset)

    redraw()
    plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else None
    run_app(path)
