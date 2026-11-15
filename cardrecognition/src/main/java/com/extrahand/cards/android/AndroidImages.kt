package com.extrahand.cards.android

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import com.extrahand.cards.core.CardImage
import com.extrahand.cards.core.NormRect
import java.io.ByteArrayOutputStream

/** Bitmap -> platform-neutral [CardImage]. */
fun Bitmap.toCardImage(): CardImage {
    val src = if (config == Bitmap.Config.HARDWARE) copy(Bitmap.Config.ARGB_8888, false) else this
    val px = IntArray(src.width * src.height)
    src.getPixels(px, 0, src.width, 0, 0, src.width, src.height)
    return CardImage(src.width, src.height, px)
}

/** [CardImage] -> Bitmap (ARGB_8888). */
fun CardImage.toBitmap(): Bitmap = Bitmap.createBitmap(argb, width, height, Bitmap.Config.ARGB_8888)

fun CardImage.toJpeg(quality: Int = 90): ByteArray = toBitmap().toJpeg(quality)

fun Bitmap.toJpeg(quality: Int = 90): ByteArray =
    ByteArrayOutputStream().also { compress(Bitmap.CompressFormat.JPEG, quality, it) }.toByteArray()

fun decodeCardImage(jpeg: ByteArray): CardImage =
    BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size).toCardImage()

/** Rotate by [degrees] (multiple of 90). */
fun Bitmap.rotated(degrees: Int): Bitmap {
    if (degrees % 360 == 0) return this
    val m = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(this, 0, 0, width, height, m, true)
}

/** Crop a Bitmap with a normalized rect. */
fun Bitmap.cropFraction(r: NormRect): Bitmap {
    val l = (r.left * width).toInt().coerceIn(0, width - 1)
    val t = (r.top * height).toInt().coerceIn(0, height - 1)
    val w = ((r.right - r.left) * width).toInt().coerceIn(1, width - l)
    val h = ((r.bottom - r.top) * height).toInt().coerceIn(1, height - t)
    return Bitmap.createBitmap(this, l, t, w, h)
}

/** Downscale so the longest side is at most [maxSide] (keeps training JPEGs small). */
fun Bitmap.limitSize(maxSide: Int): Bitmap {
    val longest = maxOf(width, height)
    if (longest <= maxSide) return this
    val s = maxSide.toFloat() / longest
    return Bitmap.createScaledBitmap(this, (width * s).toInt().coerceAtLeast(1), (height * s).toInt().coerceAtLeast(1), true)
}
