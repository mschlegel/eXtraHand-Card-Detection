# Feature extractor `luma-grid-v1` (exact spec)

Reference implementations, which must stay in sync:
`cardrecognition/src/main/java/com/extrahand/cards/core/FeatureExtractor.kt` (`LumaGridV1`) and
`tools/python/cardnet.py` (`extract`). On a lossless image they agree to about 1e-8. JPEG decoders differ
slightly between platforms, which costs a little similarity but doesn't matter for k-NN.

Input: an RGB image of one card that fills the frame (after `CardLocator`). Output: a float vector of 1248
values, L2-normalized.

1. **Orientation:** If width > height, rotate 90° clockwise.
2. **Per pixel:** Scale r, g, b to [0,1].
   * `luma = 0.299 r + 0.587 g + 0.114 b`
   * `red  = max(0, r − max(g, b) − 0.06)`
3. **Canonical size:** Box-resample both planes to 126 × 176 (W × H).
4. **Box resample** of the sub-rectangle `[x0,x1) × [y0,y1)` to `dw × dh` (integer math, floor division):
   ```
   sw = x1-x0, sh = y1-y0
   for dest column i: xa = i*sw // dw ; xb = max(xa+1, (i+1)*sw // dw)
   for dest row    j: ya = j*sh // dh ; yb = max(ya+1, (j+1)*sh // dh)
   out[j][i] = mean(src[y0+ya : y0+yb, x0+xa : x0+xb])
   ```
5. **Blocks**, computed on the canonical planes:
   * **corner(plane, gx, gy)** = average of
     * TL = resample `[0,30) × [0,52)` to gx × gy, and
     * BR = resample `[96,126) × [124,176)` to gx × gy, rotated 180° (both axes reversed).
   * **A**: `corner(luma, 16, 24)`. 384 values.
   * **B**: luma resampled from the full canonical plane to 24 × 32, then averaged with its own 180° rotation. 768 values.
   * **C**: `corner(red, 8, 12)`. 96 values.
6. **Normalize:**
   * **A** and **B**: subtract the mean and divide by the L2 norm (if the norm is below 1e-6, leave zeros), then multiply by the weight (A 1.0, B 0.6).
   * **C**: multiply by `3 / sqrt(96)`, without mean removal.
7. Concatenate them in row-major order (`index = row*gx + col`) as `A | B | C`, then L2-normalize the whole vector.

Rationale: The corner index (rank and suit glyph) carries most of the identity. The whole-card block helps
with pips and face art. The red block separates ♥♦ from ♠♣, which luma alone blurs. The 180° averaging
makes upside-down cards match.

Known limits, for a future v2: The card must be roughly axis-aligned and fill the guide (there's no
perspective correction), and recognition is sensitive to the deck design. A good v2 is a small CNN
embedding (for example a MobileNet trained on `samples/**.jpg`) under a new extractor id. The k-NN, the
dataset format, and the sync pipeline all stay as they are.
