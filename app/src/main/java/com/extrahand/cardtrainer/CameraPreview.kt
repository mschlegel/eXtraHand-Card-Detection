package com.extrahand.cardtrainer

import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size as GSize
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.extrahand.cards.core.NormRect
import java.util.concurrent.Executors

/** Upright camera frame aspect (width / height). Preview and analysis both use 4:3 so the overlay matches. */
const val FRAME_ASPECT = 3f / 4f
val GUIDE: NormRect = NormRect.centeredCard(FRAME_ASPECT, heightFraction = 0.8f)

@Composable
fun CameraPreview(analyzer: ImageAnalysis.Analyzer, torch: Boolean, modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    var camera by remember { mutableStateOf<Camera?>(null) }

    DisposableEffect(owner) {
        val executor = Executors.newSingleThreadExecutor()
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        future.addListener({
            val p = future.get().also { provider = it }
            val selector = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(
                    ResolutionStrategy(Size(1280, 960), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                ).build()
            val preview = Preview.Builder().setResolutionSelector(selector).build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build().also { it.setAnalyzer(executor, analyzer) }
            p.unbindAll()
            camera = p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis)
        }, ContextCompat.getMainExecutor(context))
        onDispose {
            provider?.unbindAll()
            executor.shutdown()
        }
    }
    LaunchedEffect(camera, torch) { camera?.cameraControl?.enableTorch(torch) }

    Box(modifier) {
        AndroidView({ previewView }, Modifier.fillMaxSize())
        GuideOverlay(Modifier.fillMaxSize())
    }
}

/** Darkens everything outside the card guide box. Geometry matches PreviewView FIT_CENTER at [FRAME_ASPECT]. */
@Composable
fun GuideOverlay(modifier: Modifier = Modifier) {
    Canvas(modifier) {
        val scale = minOf(size.width / FRAME_ASPECT, size.height)
        val fw = FRAME_ASPECT * scale
        val fh = scale
        val ox = (size.width - fw) / 2
        val oy = (size.height - fh) / 2
        val r = Rect(
            Offset(ox + GUIDE.left * fw, oy + GUIDE.top * fh),
            GSize(GUIDE.width * fw, GUIDE.height * fh),
        )
        val corner = CornerRadius(r.width * 0.05f)
        val path = Path().apply {
            fillType = PathFillType.EvenOdd
            addRect(Rect(Offset.Zero, size))
            addRoundRect(androidx.compose.ui.geometry.RoundRect(r, corner))
        }
        drawPath(path, Color.Black.copy(alpha = 0.55f))
        drawRoundRect(Color.White, r.topLeft, r.size, corner, style = Stroke(width = 3f))
        // Hint marks where the corner index should be.
        val c = GSize(r.width * 0.24f, r.height * 0.30f)
        drawRect(Color(0xFF66BB6A), r.topLeft, c, style = Stroke(width = 2f))
        drawRect(Color(0xFF66BB6A), Offset(r.right - c.width, r.bottom - c.height), c, style = Stroke(width = 2f))
    }
}
