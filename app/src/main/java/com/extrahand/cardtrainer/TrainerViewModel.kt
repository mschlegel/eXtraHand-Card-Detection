package com.extrahand.cardtrainer

import android.app.Application
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.extrahand.cards.android.limitSize
import com.extrahand.cards.android.recognize
import com.extrahand.cards.android.toBitmap
import com.extrahand.cards.android.toCardImage
import com.extrahand.cards.android.toJpeg
import com.extrahand.cards.core.Card
import com.extrahand.cards.core.DatasetCodec
import com.extrahand.cards.core.KnnCardRecognizer
import com.extrahand.cards.core.LumaGridV1
import com.extrahand.cards.core.Recognition
import com.extrahand.cards.core.SampleRecord
import com.extrahand.cards.sync.GitHubDatasetSync
import com.extrahand.cards.sync.DatasetSource
import com.extrahand.cards.sync.LocalDatasetStore
import com.extrahand.cards.sync.RemoteDatasetLoader
import com.extrahand.cardtrainer.handreader.HandReader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File
import java.util.UUID

data class Captured(
    /** Tightened card crop that will be saved as the training image. */
    val image: Bitmap,
    val recognition: Recognition,
) {
    /** What we show as "the guess": confident card, else best low-confidence candidate. */
    val guess: Card? get() = recognition.card ?: recognition.candidates.firstOrNull()?.card
    val guessConfident: Boolean get() = recognition.card != null
}

data class UiState(
    val loading: Boolean = true,
    val sampleCount: Int = 0,
    val labelsCovered: Int = 0,
    val backSamples: Int = 0,
    /** Samples labelled plain "JK" that could be sorted into high / low. */
    val unsortedJokers: Int = 0,
    val pendingCount: Int = 0,
    /** Accuracy of the last [ACCURACY_WINDOW] guesses, null if none yet. */
    val recentAccuracy: Float? = null,
    val datasetVersion: Long = 0,
    val live: Recognition? = null,
    val captured: Captured? = null,
    val picking: Boolean = false,
    val syncing: Boolean = false,
    val message: String? = null,
    val settings: TrainerSettings = TrainerSettings(),
    val showSettings: Boolean = false,
    /** Which screen is showing. Always TRAIN when the hand reader module is off. */
    val mode: Mode = Mode.TRAIN,
)

enum class Mode { TRAIN, TABLE }

const val ACCURACY_WINDOW = 50
private const val SAVE_MAX_SIDE = 480
private const val SOURCE = "card-trainer/0.1.0"

class TrainerViewModel(app: Application) : AndroidViewModel(app) {
    private val store = LocalDatasetStore(File(app.filesDir, "card-dataset"))
    private val recognizer = KnnCardRecognizer()

    /**
     * The optional hand reader module. Created on first use only, so it costs nothing while switched off.
     * It shares [recognizer], so cards (and backs) learned here are read on the table straight away.
     */
    private var _handReader: HandReader? = null
    val handReader: HandReader
        get() = _handReader ?: HandReader(recognizer, viewModelScope, _state.value.settings.multiDeck, ::localSampleImage)
            .also { _handReader = it; prefetchSamplePhotos() }

    /** A sample's photo if it is on this device (decoded at most ~480 px, as saved). */
    private fun localSampleImage(id: String, @Suppress("UNUSED_PARAMETER") label: String): com.extrahand.cards.core.CardImage? {
        val f = store.get(id)?.image?.let { store.imageFile(it) }?.takeIf { it.exists() } ?: return null
        return BitmapFactory.decodeFile(f.path)?.toCardImage()
    }

    /**
     * The hand reader matches against real photos. Samples pulled from GitHub have only their vectors on
     * the device, so download their JPEGs once (into the store's own samples/ folder) in the background.
     */
    private fun prefetchSamplePhotos() {
        viewModelScope.launch(Dispatchers.IO) {
            val missing = store.samples().filter { s -> s.image != null && !store.imageFile(s.image!!).exists() }
            if (missing.isEmpty()) return@launch
            toast("Downloading ${missing.size} sample photos for the hand reader...")
            val s = _state.value.settings
            val src = DatasetSource.GitHub(s.owner, s.repo, s.branch.ifEmpty { "main" }, s.path.trim('/'), s.token.ifBlank { null })
            val loader = RemoteDatasetLoader(src, File(getApplication<Application>().cacheDir, "sample-images"))
            var got = 0
            for (rec in missing) {
                val bytes = runCatching { loader.fetch(rec.image!!) }.getOrNull() ?: continue
                store.imageFile(rec.image!!).apply { parentFile?.mkdirs() }.writeBytes(bytes)
                got++
            }
            _handReader?.samplePhotosChanged()
            toast("Hand reader: $got of ${missing.size} sample photos downloaded")
        }
    }
    private val syncMutex = Mutex()
    @Volatile private var ready = false
    @Volatile private var latestCrop: Bitmap? = null

    private val _state = MutableStateFlow(UiState(settings = TrainerSettings.load(app)))
    val state: StateFlow<UiState> = _state

    init {
        viewModelScope.launch(Dispatchers.Default) {
            recognizer.addAll(store.samples())
            ready = true
            refreshStats()
            _state.update { it.copy(loading = false) }
            if (_state.value.settings.canSync) sync(quiet = true)
        }
    }

    // ---------- camera ----------

    /** Called from the camera analyzer thread with the guide-box crop. */
    fun onFrame(crop: Bitmap) {
        latestCrop = crop
        if (!ready || _state.value.captured != null || recognizer.sampleCount == 0) return
        val r = recognizer.recognize(crop, topN = 1)
        _state.update { if (it.captured == null) it.copy(live = r) else it }
    }

    fun capture() {
        val crop = latestCrop ?: run { toast("Camera not ready"); return }
        viewModelScope.launch(Dispatchers.Default) {
            val r = recognizer.recognize(crop.toCardImage(), topN = 5)
            val image = (r.croppedImage ?: crop.toCardImage()).toBitmap().limitSize(SAVE_MAX_SIDE)
            // Features are recomputed from the exact image we save, so the JPEG reproduces them.
            val f = LumaGridV1.extract(image.toCardImage())
            val rec = recognizer.recognizeFeatures(f, topN = 5)
            _state.update { it.copy(captured = Captured(image, rec), picking = rec.candidates.isEmpty()) }
        }
    }

    fun retake() = _state.update { it.copy(captured = null, picking = false) }
    fun showPicker(show: Boolean) = _state.update { it.copy(picking = show) }

    /** User confirmed which card this really is. Saves the sample and learns it immediately. */
    fun submit(actual: Card) {
        val cap = _state.value.captured ?: return
        _state.update { it.copy(captured = null, picking = false) }
        viewModelScope.launch(Dispatchers.IO) {
            val guess = cap.guess
            val id = GitHubDatasetSync.stamp() + "-" + UUID.randomUUID().toString().take(6)
            val features = cap.recognition.features ?: LumaGridV1.extract(cap.image.toCardImage())
            val imagePath = "samples/${actual.label}/$id.jpg"
            val record = SampleRecord(
                id = id,
                label = actual.label,
                f = DatasetCodec.encodeVector(features),
                createdAt = System.currentTimeMillis(),
                predicted = guess?.label,
                correct = guess?.let { it == actual },
                image = imagePath,
                source = "$SOURCE (${Build.MODEL})",
            )
            store.add(record, cap.image.toJpeg(90))
            recognizer.learn(id, actual, features)
            refreshStats()
            toast(if (guess == actual) "Correct: ${actual.display}" else "Learned ${actual.display}")
            maybeAutoSync()
        }
    }

    // ---------- joker sorting ----------

    /** Samples still labelled as the unspecified joker ("JK"), oldest first. */
    fun unsortedJokers(): List<SampleRecord> =
        store.samples().filter { it.label == Card.JOKER_LABEL }.sortedBy { it.createdAt }

    /** The sample's JPEG: from this device if present, else downloaded from the dataset on GitHub. Blocking. */
    fun sampleImage(rec: SampleRecord): Bitmap? {
        val path = rec.image ?: return null
        val bytes = store.imageFile(path).takeIf { it.exists() }?.readBytes() ?: runCatching {
            val s = _state.value.settings
            val src = DatasetSource.GitHub(s.owner, s.repo, s.branch.ifEmpty { "main" }, s.path.trim('/'), s.token.ifBlank { null })
            RemoteDatasetLoader(src, File(getApplication<Application>().cacheDir, "sample-images")).fetch(path)
        }.getOrNull() ?: return null
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }

    /** Re-file a sample under another card (e.g. "JK" -> high/low joker). Learned at once, published on sync. */
    fun relabel(rec: SampleRecord, to: Card) {
        viewModelScope.launch(Dispatchers.IO) {
            val newId = GitHubDatasetSync.stamp() + "-" + UUID.randomUUID().toString().take(6)
            val moved = store.relabel(rec.id, to.label, newId) ?: return@launch
            recognizer.forget(rec.id)
            recognizer.learn(moved.id, to, DatasetCodec.decodeVector(moved.f, LumaGridV1.dim))
            refreshStats()
            maybeAutoSync()
        }
    }

    fun undoLast() {
        viewModelScope.launch(Dispatchers.IO) {
            val last = store.samples().maxByOrNull { it.createdAt } ?: run { toast("Nothing to undo"); return@launch }
            store.delete(last.id)
            recognizer.forget(last.id)
            refreshStats()
            toast("Removed last sample (${Card.parse(last.label)?.display ?: last.label})")
        }
    }

    // ---------- sync ----------

    fun sync(quiet: Boolean = false) {
        val s = _state.value.settings
        if (!s.canSync) { toast("Set a GitHub token in Settings first"); showSettings(true); return }
        viewModelScope.launch(Dispatchers.IO) {
            if (!syncMutex.tryLock()) return@launch
            try {
                _state.update { it.copy(syncing = true) }
                val result = GitHubDatasetSync(store, s.gitHub()).sync { msg -> if (!quiet) toast(msg) }
                result.removedIds.forEach { recognizer.forget(it) }
                recognizer.addAll(result.importedSamples)
                refreshStats()
                val parts = buildList {
                    if (result.pushed > 0) add("pushed ${result.pushed}")
                    if (result.deleted > 0) add("deleted ${result.deleted}")
                    if (result.pulled > 0) add("pulled ${result.pulled}")
                }
                if (!quiet || parts.isNotEmpty())
                    toast("Synced v${result.datasetVersion}: " + parts.ifEmpty { listOf("up to date") }.joinToString(", "))
            } catch (e: Exception) {
                toast("Sync failed: ${e.message}")
            } finally {
                _state.update { it.copy(syncing = false) }
                syncMutex.unlock()
            }
        }
    }

    private fun maybeAutoSync() {
        val s = _state.value.settings
        if (s.canSync && s.autoSyncEvery > 0 && store.pendingCount() >= s.autoSyncEvery) sync(quiet = true)
    }

    // ---------- settings / misc ----------

    fun showSettings(show: Boolean) = _state.update { it.copy(showSettings = show) }

    fun setMode(m: Mode) = _state.update { it.copy(mode = if (it.settings.handReaderEnabled) m else Mode.TRAIN) }

    fun saveSettings(s: TrainerSettings) {
        TrainerSettings.save(getApplication(), s)
        _handReader?.setMultiDeck(s.multiDeck)
        _state.update { it.copy(settings = s, showSettings = false, mode = if (s.handReaderEnabled) it.mode else Mode.TRAIN) }
        if (s.canSync) sync()
    }

    fun clearMessage() = _state.update { it.copy(message = null) }
    private fun toast(msg: String) = _state.update { it.copy(message = msg) }

    private fun refreshStats() {
        val samples = store.samples()
        val judged = samples.filter { it.correct != null }.sortedBy { it.createdAt }.takeLast(ACCURACY_WINDOW)
        _state.update {
            it.copy(
                sampleCount = samples.size,
                labelsCovered = samples.map { s -> s.label }.toSet().count { l -> Card.parse(l)?.let { c -> !c.isBack && !c.isJoker } == true },
                backSamples = samples.count { s -> s.label == Card.BACK_LABEL },
                unsortedJokers = samples.count { s -> s.label == Card.JOKER_LABEL },
                pendingCount = store.pendingCount(),
                recentAccuracy = if (judged.isEmpty()) null else judged.count { s -> s.correct == true }.toFloat() / judged.size,
                datasetVersion = store.syncState().remoteDatasetVersion,
            )
        }
    }

}
