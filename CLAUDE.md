# CLAUDE.md: orientation for AI assistants

This repo is **part 1** of the multi-part *eXtraHand* project: playing-card recognition. It has a single
user (the repo owner), who trains the recognizer by using the Android app. Later parts consume the
trained data, either through the snap-in library or straight from GitHub.

## Read first
- `README.md`: what's here, plus the training workflow
- `docs/INTEGRATION.md`: how another app uses the recognizer (**start here if you're building part 2+**)
- `docs/DATASET_FORMAT.md`: the on-GitHub data contract
- `docs/FEATURE_SPEC.md`: the exact descriptor algorithm

## Layout
- `cardrecognition/`: Android library, namespace `com.extrahand.cards`
  - `core/`: **pure Kotlin, no Android imports** (keep it that way; it's JVM-unit-tested and portable).
    `CardRecognizer` (public interface), `KnnCardRecognizer`, `LumaGridV1`, `CardLocator`, `KnnIndex`, `Dataset.kt` (wire format), `Card` (incl. `BK` = card back).
  - `core/table/`: whole-table reading, pure Kotlin. `TableReader` (segmentation, homography rectification, corner-index
    ink search + edge scan), `CornerIndex` (corner view of luma-grid-v1 vectors), `Segmentation`, `FramePlanes`, `Geometry`,
    `SceneStabilizer`, `Rectifier`. Tested end-to-end on synthetic rendered tables in `TableReaderTest`. `RealDatasetTest` is an
    opt-in benchmark on the real dataset photos (`EXTRAHAND_REAL=1`). Run it after changing thresholds or the corner logic.
- The dataset in a fresh checkout can be stale: the app pushes sync commits to GitHub. `git fetch` before judging what is trained.
  - `sync/`: pure JVM I/O. `RemoteDatasetLoader` (read path, cached), `LocalDatasetStore` (on-device log),
    `GitHubDatasetSync` and `GitHubClient` (write path: Git Data API, one commit per sync).
  - `android/`: Bitmap adapters, the `CardRecognition` facade, and `CardFrameAnalyzer` (CameraX is `compileOnly`).
- `app/`: Card Trainer (Compose). `TrainerViewModel` owns the store, the recognizer, and sync. `CameraPreview.kt` holds the camera and the guide overlay.
  `TrainerScreen.kt` holds the UI. Settings (GitHub owner/repo/branch/path/token, module toggles) are in SharedPreferences.
  `handreader/`: the optional Hand Reader module (`HandReader` controller + `HandReaderScreen`). It is created only when the
  module is switched on and shares the trainer's recognizer.
- `dataset/`: **data, written only by the app.** Don't hand-edit or rewrite shards. They're immutable and clients cache them by name.
- `tools/python/cardnet.py`: the Python mirror of the extractor and classifier.

## Build / test
- JDK 17+ required. On this machine use Android Studio's JBR: `JAVA_HOME="/c/Program Files/Android/Android Studio/jbr"`.
- `./gradlew :cardrecognition:testDebugUnitTest`: core unit tests. They also write the parity fixture to `cardrecognition/build/parity/`.
- `./gradlew assembleDebug`: builds the APK.
- Versions: AGP 8.7.3, Kotlin 2.1.0, Gradle 8.11.1, compileSdk 35, minSdk 26, CameraX 1.4.1, Compose BOM 2024.12.01.

## Invariants (don't break these without a migration)
1. Any change to `LumaGridV1` output needs a **new extractor id**. Old vectors are incompatible. Update the Kotlin code, the Python code, and FEATURE_SPEC.md together, then rerun the parity check:
   `python -c "import numpy as np, cardnet; from PIL import Image; k=np.loadtxt('../../cardrecognition/build/parity/features.txt'); print(k @ cardnet.extract(Image.open('../../cardrecognition/build/parity/card.ppm')))"` from `tools/python`. It must print about 1.0.
2. Feature shards are immutable. Deletions use `manifest.deleted` tombstones.
3. JSON readers ignore unknown keys, so additive fields are fine.
4. `core/` and `sync/` stay free of Android APIs.
5. `CardRecognizer` is the public contract. Add methods with defaults instead of changing signatures.

## Roadmap ideas for later parts
- Card detection in a full frame (multiple cards, perspective warp). Feed the warped crops to `recognize(..., locate = false)`.
- Train a CNN or embedding model from `dataset/samples/**.jpg`. Publish it under `models/` with its own manifest, and
  ship it as a new `FeatureExtractor` (k-NN still works) or as a separate `CardRecognizer` implementation.
- Evaluate with `python tools/python/cardnet.py evaluate --dataset dataset` (leave-one-out accuracy).
