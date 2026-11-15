package com.extrahand.cards.android

import android.graphics.Bitmap
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.extrahand.cards.core.CardRecognizer
import com.extrahand.cards.core.NormRect
import com.extrahand.cards.core.Recognition

/**
 * Drop-in CameraX analyzer. Rotates each frame upright, crops the card guide box, and (optionally) runs
 * the recognizer. Requires the host app to depend on androidx.camera:camera-core (>= 1.3).
 *
 * Use with `ImageAnalysis.Builder().setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)`.
 * The guide box is expressed in normalized coordinates of the UPRIGHT analysis frame; draw the same rect
 * over a PreviewView using FIT_CENTER scale and the same aspect ratio so what the user sees is what is cropped.
 *
 * @param guide card guide box; null = use [NormRect.centeredCard] for the frame's aspect.
 * @param recognizer if null, only [onFrame] is called (useful when the app decides when to recognize).
 * @param minIntervalMs throttle recognition.
 */
class CardFrameAnalyzer(
    private val recognizer: () -> CardRecognizer?,
    private val guide: NormRect? = null,
    private val minIntervalMs: Long = 250,
    private val onFrame: (crop: Bitmap) -> Unit = {},
    private val onResult: (Recognition, crop: Bitmap) -> Unit = { _, _ -> },
) : ImageAnalysis.Analyzer {
    private var last = 0L

    override fun analyze(image: ImageProxy) {
        try {
            val now = System.currentTimeMillis()
            if (now - last < minIntervalMs) return
            last = now
            val upright = image.toBitmap().rotated(image.imageInfo.rotationDegrees)
            val rect = guide ?: NormRect.centeredCard(upright.width.toFloat() / upright.height)
            val crop = upright.cropFraction(rect)
            onFrame(crop)
            recognizer()?.let { onResult(it.recognize(crop), crop) }
        } finally {
            image.close()
        }
    }
}
