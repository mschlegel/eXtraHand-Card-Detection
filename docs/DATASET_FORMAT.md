# Dataset format (formatVersion 1)

Root: `dataset/` in `mschlegel/eXtraHand-Card-Detection`, branch `main`.
Public raw base URL: `https://raw.githubusercontent.com/mschlegel/eXtraHand-Card-Detection/main/dataset/`

```
dataset/
  manifest.json                     the only mutable file; read it first
  features/000001-20261115T201500Z.json   immutable feature shards, one per sync
  features/000002-....json
  samples/<LABEL>/<sampleId>.jpg    card crop each sample came from (source of truth)
```

Only the Card Trainer app writes here, through `GitHubDatasetSync`. It makes one commit per sync, with a
message like `dataset v12: +10 samples (240 total)`. Don't hand-edit shards. To fix a bad label, delete the
sample in the app, or add a tombstone and a corrected sample.

## manifest.json

```json
{
  "formatVersion": 1,
  "datasetVersion": 12,                 // +1 on every published change
  "updatedAt": "2026-11-15T20:15:00Z",
  "extractor": "luma-grid-v1",          // MUST match your feature extractor
  "featureDim": 1248,
  "encoding": "int8-base64",
  "sampleCount": 240,                   // live samples (after tombstones)
  "labelCounts": { "AS": 5, "10H": 4 },
  "shards": [ { "file": "features/000001-20261115T201500Z.json", "count": 10, "sha256": "…" } ],
  "deleted": [ "20261115T201233Z-a1b2c3" ]   // tombstoned sample ids
}
```

Loading algorithm:
1. GET `manifest.json`. A 404 means nothing is published yet, so treat it as empty.
2. Check that `extractor` and `featureDim` match yours. If they don't, rebuild features from the JPEGs.
3. For each `shards[i]`, use the cached copy if one exists with the same file name and `sha256`.
   Otherwise GET it. Shards never change after they are written, so caching by name is safe.
4. Concatenate all `samples`, in shard order. Drop ids in `deleted` and duplicate ids (keep the first).

## features/*.json (FeatureShard)

```json
{
  "formatVersion": 1, "extractor": "luma-grid-v1", "featureDim": 1248, "encoding": "int8-base64",
  "samples": [
    {
      "id": "20261115T201233Z-a1b2c3",   // unique; UTC timestamp + random
      "label": "10H",                    // see Labels
      "f": "base64…",                    // 1248 signed int8 values
      "createdAt": 1763237553000,        // epoch ms
      "predicted": "10D",                // guess shown to the user before labelling (optional)
      "correct": false,                  // whether that guess was right (optional)
      "image": "samples/10H/20261115T201233Z-a1b2c3.jpg",   // relative to dataset root (optional)
      "source": "card-trainer/0.1.0 (Pixel 8)"
    }
  ]
}
```

### Vector encoding `int8-base64`

`f` is base64 of `featureDim` bytes, each a signed int8. The vector is `v / max|v| * 127`, rounded and
clamped to ±127. To decode: cast to float, then L2-normalize. Similarity is the dot product (cosine).
The scale is discarded on purpose, since cosine doesn't depend on it.

### Labels

`<rank><suit>`. Rank: `A 2 3 4 5 6 7 8 9 10 J Q K`. Suit: `S H D C`. Examples: `AS`, `10H`, `QD`.
The joker is `JK`. Parsers should also accept `T` for ten and lowercase.

### Images

Images are JPEG (quality 90), with the longest side at most 480 px. Each is the tightened card crop the
features were computed from, held upright or upside-down as captured. This is the data to use when
training a real model later.

## Classification (reference)

The reference classifier is k-NN with k = 7. Each neighbour gets the weight `exp((sim - sim_best) / 0.03)`,
and the weights are summed per label, then normalized to probabilities. A card is reported only when the
probability is at least 0.5 and the best similarity is at least 0.75. Otherwise the result counts as
"unsure", and candidates are still returned.

## Versioning rules for future parts

* Additive JSON fields are allowed. Readers must ignore unknown keys.
* A new feature algorithm needs a new `extractor` id. Recompute it from `samples/**.jpg` and write it as
  a separate dataset root (for example `dataset-v2/`), or rewrite all shards in one migration commit and bump `formatVersion`.
* Trained models (TFLite/ONNX), when they exist, go under `models/<name>/` with their own manifest. They
  don't go in `dataset/`.
