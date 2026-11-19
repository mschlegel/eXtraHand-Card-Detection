# eXtraHand Card Detection

Part 1 of the eXtraHand project: an Android **training app** for playing-card recognition, a reusable
**snap-in recognition library**, and the **trained dataset** itself. All three live in this repo, and the
app publishes new training data back to the repo.

```
Show card to camera ─► app guesses ─► you tap ✓ Correct or ✗ Wrong + pick the real card
                                         │
                              saved + learned instantly on the phone
                                         │
                               Sync ─► commit to GitHub  dataset/
                                         │
               any app ◄── CardRecognition.load()  /  cardnet.py  /  raw URL
```

| Path | What |
|---|---|
| `app/` | **Card Trainer** Android app (Kotlin, Compose, CameraX) |
| `cardrecognition/` | **Snap-in library** (`com.extrahand.cards`): `CardRecognizer` interface, feature extractor, k-NN, loader, GitHub sync |
| `dataset/` | **The trained data.** Written only by the app's Sync. `manifest.json` + immutable `features/*.json` shards + `samples/<LABEL>/*.jpg` |
| `tools/python/cardnet.py` | Reference Python loader/classifier (bit-exact with the Kotlin extractor) |
| `docs/` | Integration guide, dataset format, feature spec |
| `CLAUDE.md` | Orientation for AI assistants working on later parts |

## Using the trained data from another app

Android/Kotlin (see [docs/INTEGRATION.md](docs/INTEGRATION.md)):

```kotlin
val recognizer = CardRecognition.load(context)    // newest data from this repo, cached, works offline
val result = recognizer.recognize(cardBitmap)
result.card?.label                                // "QH" or null when unsure
```

Anything else: read `https://raw.githubusercontent.com/mschlegel/eXtraHand-Card-Detection/main/dataset/manifest.json`,
then each shard it lists. Format: [docs/DATASET_FORMAT.md](docs/DATASET_FORMAT.md). Python: `tools/python/cardnet.py`.

## Building the app

Requirements: Android Studio (or JDK 17+ and Android SDK 35).

```bash
./gradlew assembleDebug
```

APK: `app/build/outputs/apk/debug/app-debug.apk`. Install with `adb install -r app/build/outputs/apk/debug/app-debug.apk`,
or open the project in Android Studio and press Run.

```bash
./gradlew :cardrecognition:testDebugUnitTest
```

## Training workflow

1. First launch: grant camera access. Then open **⚙** and paste a GitHub token. Use a fine-grained PAT
   limited to this repo with **Contents: Read and write**.
2. Hold a card inside the white box, with the corner index in the green box (upside-down works too). Tap **Capture**.
3. If the guess is right, tap **✓ Correct**. If it's wrong, tap **✗ Wrong**, pick the rank and suit, then tap **Save**.
   You can also tap one of the "Or:" alternatives.
4. Every sample is learned immediately. The app syncs automatically every 10 changes (configurable), and **Sync** pushes now.
   **Undo** removes the most recent sample (it is also deleted remotely on the next sync).
5. The top line shows samples, cards covered (of 52), unsynced changes, accuracy over the last 50 guesses, and the dataset version.

Tips: Use even light (the **Light** button turns on the torch) on a plain, darker background. A few samples
per card under different lighting beat many identical ones.

**Jokers:** use the **High joker** / **Low joker** chips (big / small). **Joker (either)** is for decks whose
two jokers are identical. Samples saved before this split show up as "N joker samples not marked high/low ·
Sort". Tap it to file each photo as High or Low. It's published on the next sync like any other change.

**Card backs:** to detect face-down cards (draw pile, opponents' hands), capture the back of your deck a few
times. Tap **✗ Wrong** and then the **Back** chip. Backs are stored under the label `BK`.

## Hand reader module (optional)

Turn it on in **⚙ → Modules → Hand reader**. A **Table** button then switches to a full-frame camera that
reads everything it can see, with the camera anywhere around the table:

* fanned, spread or overlapping hands, single cards (rotated, at an angle, far away) and loose groups,
* face-down cards: a single back or squared pile (e.g. the draw deck), or a spread of backs (with a lower
  bound on the count).

It needs no new training. It matches against your sample photos (any that are only on GitHub are
downloaded once in the background when you first open it). Fully visible cards are rectified and matched with the normal descriptor, and
partly hidden cards are matched by their corner index, which is part of every existing sample. **Live**
reads about once a second and smooths the results. **Read table (full res)** takes one full-resolution
photo and reads it, which works best for distant cards (use the zoom slider too). Turn on **Multiple
decks** for games where a card can appear twice. Scoring a hand is left to later parts. The library
returns groups of cards (`core.table.TableReader`, see [docs/INTEGRATION.md](docs/INTEGRATION.md)).

## How recognition works (v1)

There's no neural network yet. Each card crop is turned into a 1248-number descriptor (`luma-grid-v1`:
the corner index, a coarse view of the whole card, and the red ink in the corner), then classified by
k-nearest-neighbour against every labelled sample. This means:

* it learns instantly from each correction, with no training step,
* the format is simple enough to reimplement on any platform (done in Python),
* the JPEGs are kept, so a future part can train a real CNN/YOLO model on the same data, and
  publish it next to (or instead of) the k-NN shards. See [CLAUDE.md](CLAUDE.md) for the roadmap notes.
