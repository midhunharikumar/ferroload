# Speeding up image decode in Ferroload — research + recommendations

**Context.** The 3-way benchmark ([REPORT.md](REPORT.md)) showed image decode is the
bottleneck for JPEG-bound datasets, and apples-to-apples (same `DataLoader(nw=8)`)
Ferroload's JPEG throughput is ~on par with HF/PIL and behind WebDataset. The
`turbojpeg` feature brought per-core decode to ≈ PIL/libjpeg-turbo. This note
researches how to go *faster than* that.

## ✅ Implemented & measured: #1 `fast_image_resize` (SIMD resize)

`decode_resized` now decodes to RGB once (libjpeg-turbo via `turbojpeg`, else
zune-jpeg) and resizes with **`fast_image_resize`** (NEON on ARM, AVX2 on x86)
instead of the scalar `image::imageops::resize`. Measured on **M2 Pro (ARM64)**,
`ferro_native` (1 process), samples/s:

| dataset | before (scalar resize) | after (SIMD resize) | Δ | HF nw=8 | WDS nw=8 |
|---|--:|--:|--:|--:|--:|
| Stanford-Cars — zune+fir | 5,359 | **7,900** | +47% | 5,927 | 9,519 |
| Stanford-Cars — turbojpeg+fir | 6,356 | **~10,000** | +57% | 5,927 | 9,519 |
| FFHQ-256 — zune+fir | 5,223 | **7,600** | +45% | 4,887 | 8,148 |
| FFHQ-256 — turbojpeg+fir | 6,139 | **~9,600** | +57% | 4,887 | 8,148 |

**Why so large?** The *resize* was a hidden ~35% of `decode_resized`; SIMD resize
(~7× on NEON) cut it, so total throughput rose ~45–57%. Net: **Ferroload native now
*beats* WebDataset nw=8 on JPEG** (turbojpeg+fir) — and even the **pure-Rust default
(zune+fir) beats HF nw=8** — from a single process. (Confirming the diagnosis that
HF's edge was Pillow's C resize + libjpeg-turbo, not a better decoder.)

**Caveat — same-DataLoader apples-to-apples is unchanged (~5.5k):** in the worker
DataLoader at nw=8 the win washes out, because that path is IPC-bound (moving
decoded tensors worker→main), not decode-bound. So the gain is fully realized only
in the **native** in-process path — another reason to run Ferroload native.

Verified correct (`test_map.py`, `test_combinations.py`). `fast_image_resize` is
pure-Rust + cross-platform (no new C dep); it helps both the zune and turbojpeg
decoders. **Next:** decode-at-scale (#2/§Tier-1.2) and the GPU path (#5).

## What the literature says (2025–2026)

**A Rust decoder is already the fastest CPU JPEG decoder on Apple Silicon.** The
Jan-2025 "Need for Speed" benchmark of JPEG decoders measured, on an **M4 Max
(ARM64)**: **kornia-rs 1034 img/s** > OpenCV 1016 > torchvision 992 > imageio 777 >
Pillow 775 > TensorFlow 664. On x86_64 the leaders were jpeg4py, kornia-rs, OpenCV.
Two conclusions the paper draws:
- **The backend dominates.** Libraries that bind **libjpeg-turbo directly**
  (jpeg4py, kornia-rs, OpenCV, torchvision) substantially beat standard-libjpeg
  users (Pillow, imageio, scikit-image).
- **Abstraction layers hurt.** "Multiple abstraction layers tend to impact
  performance negatively." kornia-rs wins partly by being lean Rust over
  libjpeg-turbo.

So Ferroload (Rust + `turbojpeg`) is on the *right* path — the gap to "fastest" is
about leanness and a couple of missing optimizations, not the language.

`zune-jpeg` (the pure-Rust backend the `image` crate uses, Ferroload's default) is
"within ±10 ms of libjpeg-turbo" — near-parity, so the default isn't bad; the
`turbojpeg` feature closes the remainder.

## Cross-platform is a hard requirement (mac + linux, x86 + ARM)

Decode must be fast on **macOS-arm64** (Apple Silicon), **Linux-x86_64** (most
training/cloud), and **Linux-aarch64** (Graviton/Ampere) — macOS-x86 still works
but is fading. The deciding factor is which libraries have SIMD on **both** x86
(SSE/AVX2) **and** ARM (NEON):

| approach | x86 SIMD | ARM/NEON SIMD | mac | linux | native build cost |
|---|---|---|---|---|---|
| **zune-jpeg** (pure-Rust default) | ✅ AVX2/SSE | ⚠️ docs say "AVX2/SSE"; **empirically only ~18% behind libjpeg-turbo on M2 (ARM)** — fine | ✅ | ✅ | none |
| **libjpeg-turbo** (`turbojpeg`) | ✅ SSE2/AVX2 | ✅ **NEON** (2–6× vs libjpeg on ARM too) | ✅ | ✅ | C lib per target (x86 SIMD needs nasm; ARM uses NEON intrinsics — the `turbojpeg` crate vendors both) |
| **fast_image_resize** (resize) | ✅ AVX2 (14–23×) | ✅ **NEON (~7×)** | ✅ | ✅ | none (pure Rust) |
| **nvJPEG / nvImageCodec** (GPU) | NVIDIA x86 | NVIDIA ARM (Grace/Jetson) | ❌ no CUDA on mac | ✅ NVIDIA only | CUDA toolkit |
| **Apple ImageIO** | — | mac-arm only | ✅ mac only | ❌ | mac-only, no clean Rust binding |

**Reading of the matrix:**
- **`fast_image_resize` is a free win on all four targets** (NEON + AVX2). Use it
  regardless of decoder — universal.
- **libjpeg-turbo is the only JPEG decoder with confirmed SIMD on both x86 and ARM**
  — so the `turbojpeg` path is the **cross-platform-fastest**, at the cost of a C
  build per target. (My own ARM measurement: turbojpeg beats zune by ~18% on M2.)
- **The pure-Rust default (zune-jpeg) stays the portable baseline** — it builds the
  rustls-style portable wheel everywhere with no C deps, and is empirically within
  ~18% of libjpeg-turbo on ARM too (not scalar-slow). Keep it the default; make
  `turbojpeg` the opt-in "max speed, accepts a native dep" build.
- **GPU and Apple paths are platform-specific accelerators, not the cross-platform
  baseline** — they help one target each, so they're additive, never the answer to
  "fast everywhere."

## The biggest wins (mapped to Ferroload code)

### Tier 1 — low-risk pure-Rust wins (do these first)

1. **Replace `image::imageops::resize` with [`fast_image_resize`](https://github.com/Cykooz/fast_image_resize).**
   `decode_resized` currently decodes then resizes via `image::imageops::resize`,
   which is *not* SIMD-optimized. `fast_image_resize` is **14–23× faster** (bilinear
   ~23×, Lanczos3 ~14× vs the `image` crate) with SSE4.1/AVX2 **and ARM/NEON**.
   Since the benchmark transform is decode→resize-to-224, the resize is a real
   slice of the per-sample cost — this speeds up *both* the zune and turbojpeg
   paths, no C deps. **Highest value / lowest risk.**
   *Change:* `crates/ferroload-codec/src/image_codec.rs` `decode_resized` →
   `fast_image_resize::Resizer` instead of `imageops::resize`.

2. **Decode-at-scale (libjpeg-turbo DCT scaling).** libjpeg-turbo can IDCT-decode
   directly at **1/2, 1/4, 1/8** resolution (`tj3SetScalingFactor` / the `turbojpeg`
   `Decompressor` scaling factors), skipping pixels you're about to throw away.
   When source ≫ target (FFHQ-1024→224, laion variable→224, any "decode big JPEG,
   train at 224") this is a large win — fewer IDCT ops and it can skip Huffman work
   for high-frequency coefficients. (torchvision is adding the same: "resize during
   decode", pytorch/vision #8986.) *Change:* in the `turbojpeg` path, read the JPEG
   header, pick the smallest libjpeg-turbo scale ≥ target, decode at that scale,
   then `fast_image_resize` to exact. Most impactful on **large** source images
   (our FFHQ-256/cars-256 sources are already small, so measure on 512px+).

3. **Lean the turbojpeg path.** Current `decode_resized` does
   `turbojpeg → image::RgbImage::from_raw → imageops::resize` — an avoidable
   round-trip. Decode straight into the resizer's input buffer (the "abstraction
   layers hurt" finding). Minor but free.

### Tier 2 — bigger / architectural

4. **GPU decode (nvJPEG / nvImageCodec / DALI).** On NVIDIA training boxes, offload
   JPEG decode to the GPU: NVIDIA's hardware JPEG decoder (Ampere+/A100) via
   **nvImageCodec**/**nvJPEG**, as used by **DALI** (1.3–3× single-GPU, 3–10×
   multi-GPU sharing a host) and `torchvision.io.decode_jpeg(device="cuda")`. This
   is the **largest absolute win** for the common case (training on NVIDIA), and it
   keeps the CPU free. Cost: a CUDA dependency + platform gating; expose a
   `decode_many(device="cuda")` that returns GPU tensors (or a DALI-style external
   source). Worth it given most training is on NVIDIA.

5. **Offline pre-decode/resize as an enrichment layer (Ferroload-native superpower).**
   Use `Dataset.map` to decode+resize **once** and store the resulting uint8/fp16
   tensors as a new modality; training then reads with **zero decode** — turning the
   decode-bound regime into the IO-bound regime where Ferroload already wins (tiny
   reads, coalesced, in-process). Best when you run many epochs over a fixed
   resolution. Neither WebDataset nor HF Arrow has this "materialize a derived
   tensor column over the same dataset" story. Mostly docs + a recipe; the machinery
   exists.

### Tier 3 — situational

6. **Apple Silicon hardware decode (macOS only).** VideoToolbox is **video-only**
   (H.264/HEVC/ProRes/AV1, no JPEG); `ImageIO` does JPEG with Accelerate but has no
   clean Rust binding (would need `objc2`/CoreFoundation). Low priority — dev
   machines, not training fleets.
7. **Pipeline overlap.** Hide decode behind GPU compute via prefetch (the loader
   already prefetches; the micro-benchmark measures decode in isolation).

## Recommended order (cross-platform: mac + linux, x86 + ARM)

1. **`fast_image_resize` in `decode_resized`** — quick, pure-Rust, SIMD on **both**
   AVX2 (14–23×) and NEON (~7×). Helps every JPEG run on every target. **Start here.**
2. **Keep zune-jpeg the default; make `turbojpeg` the opt-in "fast everywhere" build.**
   Ship two wheel flavors: the portable pure-Rust default (no C deps, builds on all
   4 targets), and a `turbojpeg` build for max speed (libjpeg-turbo NEON+AVX2). Make
   sure the ARM `turbojpeg` build actually links NEON-enabled libjpeg-turbo.
3. **Decode-at-scale** in the `turbojpeg` path (DCT 1/2,1/4,1/8) — cross-platform
   (libjpeg-turbo has it on x86 and ARM); big on large source images.
4. **Pre-decode `map` layer** for many-epoch training — platform-agnostic; zero
   decode at train time.
5. **GPU decode** (`decode_many(device="cuda")` via nvImageCodec) — NVIDIA-only
   accelerator (Linux x86/ARM); the biggest absolute win on NVIDIA fleets, additive
   to the cross-platform CPU baseline above.

Tiers 1–2 plausibly push Ferroload's CPU JPEG decode to the **top of the CPU pack**
(kornia-rs territory) and ahead of HF/PIL; the remaining gap to WebDataset in the
benchmark was its sequential-streaming access pattern (+ no IPC), addressed
separately by running Ferroload **native** and by the GCS coalescing fix.

## Sources

- [Need for Speed: A Comprehensive Benchmark of JPEG Decoders in Python (arXiv 2501.13131)](https://arxiv.org/html/2501.13131v1)
- [libjpeg-turbo — Performance](https://libjpeg-turbo.org/About/Performance) · [DCT scaling / SmartScale](https://libjpeg-turbo.org/About/SmartScale) · [TurboJPEG API](https://deepwiki.com/libjpeg-turbo/libjpeg-turbo/4-turbojpeg-api)
- [pytorch/vision #8986 — resize during JPEG decode](https://github.com/pytorch/vision/issues/8986)
- [Cykooz/fast_image_resize (SIMD resize, benchmarks)](https://github.com/Cykooz/fast_image_resize/blob/main/benchmarks-x86_64.md)
- [kornia-rs (Rust CV, libjpeg-turbo backend)](https://github.com/kornia/kornia-rs) · [zune-jpeg](https://lib.rs/crates/zune-jpeg) · [turbojpeg crate](https://lib.rs/crates/turbojpeg)
- [NVIDIA DALI + hardware JPEG decoder (A100)](https://developer.nvidia.com/blog/loading-data-fast-with-dali-and-new-jpeg-decoder-in-a100/) · [NVIDIA DALI](https://github.com/nvidia/dali)
