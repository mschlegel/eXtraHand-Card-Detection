# Integrating the card recognizer (snap-in module)

## 1. Add the dependency

**Option A: JitPack** (no local checkout needed). In `settings.gradle.kts`:

```kotlin
dependencyResolutionManagement {
    repositories { google(); mavenCentral(); maven("https://jitpack.io") }
}
```

In your module's dependencies:

```kotlin
implementation("com.github.mschlegel:eXtraHand-Card-Detection:<git tag or commit sha>")
// Only if you use CardFrameAnalyzer:
implementation("androidx.camera:camera-core:1.4.1")
```

**Option B: Source module.** Copy or submodule this repo, then in `settings.gradle.kts`:

```kotlin
include(":cardrecognition")
project(":cardrecognition").projectDir = file("../eXtraHand-Card-Detection/cardrecognition")
```

Add `implementation(project(":cardrecognition"))`.

Requirements: minSdk 26, Java 17. The library already declares `INTERNET`.

## 2. Load and recognize

```kotlin
import com.extrahand.cards.android.CardRecognition
import com.extrahand.cards.android.recognize      // Bitmap extension

lifecycleScope.launch {
    val recognizer = CardRecognition.load(context) // suspend; downloads the newest dataset, caches it
    val r = recognizer.recognize(bitmap)           // bitmap: one card roughly filling the image
    when {
        r.card != null -> show("${r.card.display} ${r.confidence}")      // confident
        r.candidates.isNotEmpty() -> show("maybe ${r.candidates[0].card.display}")
        else -> show("no data / no idea")
    }
}
```

* `CardRecognition.load(context, source)`: `source` defaults to `DatasetSource.DEFAULT` (this repo, public, `main`, `dataset/`).
  * Private repo or always-fresh reads: `DatasetSource.GitHub(owner, repo, branch, path, token = "...")`.
  * Self-hosted copy: `DatasetSource.Http("https://host/path/dataset")`.
* Offline: if the network fails, the last cached copy is used. `CardRecognition.loadCached(context)` never touches the network.
* Refresh: call `load` again (for example on app start). It costs one small `manifest.json` request. Shards are immutable and downloaded once.
* `raw.githubusercontent.com` is CDN-cached for about 5 minutes, so new training can take that long to appear. Pass a token to bypass the cache.
* Thresholds: `CardRecognition.load(context, thresholds = RecognitionThresholds(minProbability = 0.6f, minSimilarity = 0.8f))`.

## 3. Camera input

Recognition expects a crop of one card. The simplest path is a fixed on-screen guide box. That's what
the trainer does, and `CardFrameAnalyzer` packages it:

```kotlin
val guide = NormRect.centeredCard(frameAspect = 3f / 4f, heightFraction = 0.8f)
val analysis = ImageAnalysis.Builder()
    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
    .setResolutionSelector(/* 4:3, same as Preview */)
    .build()
analysis.setAnalyzer(executor, CardFrameAnalyzer(
    recognizer = { recognizer },        // lambda so you can swap in a refreshed recognizer
    guide = guide,
    onResult = { result, crop -> /* off main thread */ },
))
```

Use `PreviewView.ScaleType.FIT_CENTER` with the same 4:3 aspect, and draw `guide` over the preview so the
user sees exactly what is cropped. See `app/.../CameraPreview.kt` (`GuideOverlay`).

`recognize()` tightens the crop to the bright card face (`CardLocator`) by default. Pass `locate = false` if
you already have a tight crop, for example from your own detector.

## 4. API surface (package `com.extrahand.cards`)

| Type | Purpose |
|---|---|
| `core.CardRecognizer` | **The interface.** `recognize(CardImage)`, `recognizeFeatures(FloatArray)`, `sampleCount`, `labelCounts()`, `extractorId` |
| `core.TrainableCardRecognizer` | Adds `learn(id, card, features)`, `forget(id)`, `clear()` |
| `core.KnnCardRecognizer` | Default implementation. `KnnCardRecognizer.fromDataset(manifest, shards)` for plain JVM |
| `core.Recognition` / `Candidate` | Result: `card?`, `confidence`, `similarity`, `candidates`, `features`, `croppedImage` |
| `core.Card`, `Rank`, `Suit` | `Card.parse("10H")`, `.label`, `.display` ("10♥"), `Card.STANDARD_DECK`, `Card.JOKER` |
| `core.CardImage`, `NormRect` | Platform-neutral ARGB image; normalized rects; `NormRect.centeredCard()` |
| `core.FeatureExtractor`, `LumaGridV1` | Descriptor (see FEATURE_SPEC.md) |
| `core.CardLocator` | Crop tightening heuristic |
| `core.DatasetCodec`, `DatasetManifest`, `FeatureShard`, `SampleRecord` | Wire format (see DATASET_FORMAT.md) |
| `android.CardRecognition` | `load` / `loadDetailed` / `loadCached` |
| `android.CardFrameAnalyzer` | CameraX analyzer (guide crop and recognize) |
| `android.*` extensions | `Bitmap.toCardImage()`, `CardImage.toBitmap()`, `toJpeg()`, `rotated()`, `cropFraction()` |
| `sync.RemoteDatasetLoader` | Cached downloader (pure JVM) |
| `sync.LocalDatasetStore`, `GitHubDatasetSync`, `GitHubClient` | Write path used by the trainer (needs a token) |

Threading: `recognize` is CPU-only and thread-safe (about 1–5 ms per 1000 samples). `load` is a suspend
function that runs on IO. `GitHubDatasetSync.sync()` blocks.

## 5. Non-Android consumers

* **JVM (desktop/server Kotlin):** The `core` and `sync` packages have no Android imports. Use
  `RemoteDatasetLoader(DatasetSource.DEFAULT, cacheDir).load()` then `KnnCardRecognizer.fromDataset(...)`.
  Convert images to `CardImage(width, height, argbIntArray)` yourself.
* **Python:** `tools/python/cardnet.py` has `Dataset.load()`, `.classify(PIL.Image)`, `extract()`, and `locate()`.
* **Other (JS, C#, Unity…):** Implement FEATURE_SPEC.md, which is about 80 lines, then fetch per DATASET_FORMAT.md.
  Check parity with the fixture written by `ParityFixtureTest` (`cardrecognition/build/parity/`).
