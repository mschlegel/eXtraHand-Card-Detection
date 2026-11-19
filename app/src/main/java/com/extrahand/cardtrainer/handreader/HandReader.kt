package com.extrahand.cardtrainer.handreader

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import com.extrahand.cards.android.limitSize
import com.extrahand.cards.android.rotated
import com.extrahand.cards.android.toCardImage
import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.KnnCardRecognizer
import com.extrahand.cards.core.table.SceneStabilizer
import com.extrahand.cards.core.table.TableReader
import com.extrahand.cards.core.table.TableReaderConfig
import com.extrahand.cards.core.table.TableScene
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import java.util.concurrent.Executor

/**
 * The hand reader module: runs [TableReader] on live camera frames (stabilized) and on full-resolution
 * snapshots. It owns no camera. [HandReaderScreen] binds the camera and hands frames to [analyzer] and
 * [snapshot]. It reads with the trainer's own recognizer, so new training applies at once.
 */
class HandReader(
    private val recognizer: KnnCardRecognizer,
    private val scope: CoroutineScope,
    multiDeck: Boolean,
    /** Decoded JPEG of a training sample (id, label), or null. Lets the reader match against real photos. */
    sampleImages: ((id: String, label: String) -> CardImage?)? = null,
) {
    data class Snapshot(val image: Bitmap, val scene: TableScene)

    data class State(
        /** Smoothed result of the live camera feed (null until the first frame is read). */
        val live: TableScene? = null,
        /** A frozen full-resolution read; while set, live reading pauses. */
        val snapshot: Snapshot? = null,
        val liveOn: Boolean = true,
        val busy: Boolean = false,
        val error: String? = null,
    )

    private val _state = MutableStateFlow(State())
    val state: StateFlow<State> = _state

    private val reader = TableReader(recognizer, TableReaderConfig(multiDeck = multiDeck), sampleImages)
    private val stabilizer = SceneStabilizer()

    fun setMultiDeck(on: Boolean) {
        if (reader.config.multiDeck == on) return
        reader.config = reader.config.copy(multiDeck = on)
        stabilizer.reset()
    }

    fun setLive(on: Boolean) {
        if (!on) stabilizer.reset()
        _state.update { it.copy(liveOn = on, live = if (on) it.live else null) }
    }

    /** More sample photos are on the device now (e.g. a download finished): rebuild the photo index. */
    fun samplePhotosChanged() = reader.invalidate()

    fun clearSnapshot() = _state.update { it.copy(snapshot = null, error = null) }

    /** CameraX analyzer for live reading. Frames arriving while a read is running are dropped by CameraX. */
    val analyzer = ImageAnalysis.Analyzer { image: ImageProxy ->
        try {
            val s = _state.value
            if (!s.liveOn || s.snapshot != null || s.busy || recognizer.sampleCount == 0) return@Analyzer
            val frame = image.toBitmap().rotated(image.imageInfo.rotationDegrees)
            val scene = stabilizer.update(reader.read(frame.toCardImage()))
            _state.update { if (it.liveOn && it.snapshot == null) it.copy(live = scene) else it }
        } catch (e: Exception) {
            _state.update { it.copy(error = "Live read failed: ${e.message}") }
        } finally {
            image.close()
        }
    }

    /** Take one full-resolution picture and read it: the best option for far-away cards. */
    fun snapshot(capture: ImageCapture, executor: Executor) {
        if (_state.value.busy) return
        _state.update { it.copy(busy = true, error = null) }
        capture.takePicture(executor, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                val bmp = try {
                    image.toBitmap().rotated(image.imageInfo.rotationDegrees).limitSize(SNAPSHOT_MAX_SIDE)
                } finally { image.close() }
                scope.launch(Dispatchers.Default) {
                    try {
                        val scene = reader.read(bmp.toCardImage())
                        _state.update { it.copy(snapshot = Snapshot(bmp, scene), busy = false) }
                    } catch (e: Throwable) {
                        _state.update { it.copy(busy = false, error = "Read failed: ${e.message}") }
                    }
                }
            }

            override fun onError(e: ImageCaptureException) {
                _state.update { it.copy(busy = false, error = "Capture failed: ${e.message}") }
            }
        })
    }

    companion object {
        /** Longest side of snapshot frames. 12 MP gains little over this and costs a lot of memory. */
        const val SNAPSHOT_MAX_SIDE = 3264
    }
}
