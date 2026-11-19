package com.extrahand.cardtrainer.handreader

import android.util.Size
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextMeasurer
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.extrahand.cards.core.table.BackLayout
import com.extrahand.cards.core.table.Pt
import com.extrahand.cards.core.table.Quad
import com.extrahand.cards.core.table.ReadMethod
import com.extrahand.cards.core.table.TableScene
import java.util.concurrent.Executors

private val FullCardColor = Color(0xFF66BB6A)
private val CornerColor = Color(0xFF4FC3F7)
private val BackColor = Color(0xFFFFB74D)

@Composable
fun HandReaderScreen(reader: HandReader, torch: Boolean, sampleCount: Int, backSamples: Int, modifier: Modifier = Modifier) {
    val s by reader.state.collectAsState()
    var capture by remember { mutableStateOf<ImageCapture?>(null) }
    val captureExecutor = remember { Executors.newSingleThreadExecutor() }
    DisposableEffect(Unit) { onDispose { captureExecutor.shutdown() } }

    Column(modifier) {
        Box(Modifier.weight(1f).fillMaxWidth().background(Color.Black)) {
            val snap = s.snapshot
            if (snap == null) {
                TableCamera(reader.analyzer, torch, onCapture = { capture = it }, Modifier.fillMaxSize())
                s.live?.let { SceneOverlay(it, Modifier.fillMaxSize()) }
            } else {
                Image(snap.image.asImageBitmap(), "snapshot", Modifier.fillMaxSize(), contentScale = ContentScale.Fit)
                SceneOverlay(snap.scene, Modifier.fillMaxSize())
            }
            if (s.busy) CircularProgressIndicator(Modifier.align(Alignment.Center))
            val hint = when {
                sampleCount == 0 -> "No training data yet. Train some cards first."
                snap == null && !s.liveOn -> "Live reading paused"
                else -> null
            }
            hint?.let { Badge(it, Modifier.align(Alignment.TopCenter).padding(8.dp)) }
        }
        Results(s.snapshot?.scene ?: s.live, backSamples, s.error)
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp), horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            if (s.snapshot == null) {
                Button(
                    onClick = { capture?.let { reader.snapshot(it, captureExecutor) } },
                    enabled = capture != null && !s.busy && sampleCount > 0,
                    modifier = Modifier.weight(2f).height(52.dp),
                ) { Text("Read table (full res)") }
                OutlinedButton(onClick = { reader.setLive(!s.liveOn) }, modifier = Modifier.weight(1f).height(52.dp)) {
                    Text(if (s.liveOn) "Pause live" else "Live")
                }
            } else {
                Button(onClick = reader::clearSnapshot, modifier = Modifier.weight(1f).height(52.dp)) { Text("Back to camera") }
            }
        }
    }
}

@Composable
private fun Badge(text: String, modifier: Modifier) = Text(
    text, color = Color.White, fontSize = 14.sp,
    modifier = modifier.background(Color(0xAA000000), MaterialTheme.shapes.small).padding(horizontal = 12.dp, vertical = 6.dp),
)

/** Text summary of a scene: one line per hand / group, then face-down cards. */
@Composable
private fun Results(scene: TableScene?, backSamples: Int, error: String?) {
    Column(
        Modifier.fillMaxWidth().heightIn(max = 180.dp).verticalScroll(rememberScrollState()).padding(horizontal = 12.dp, vertical = 6.dp),
        verticalArrangement = Arrangement.spacedBy(4.dp),
    ) {
        error?.let { Text(it, color = Color(0xFFEF5350), fontSize = 13.sp) }
        if (scene == null) { Text("Point the camera at the table.", color = Color.LightGray); return@Column }
        if (scene.isEmpty) Text("No cards seen.", color = Color.LightGray)
        scene.groups.forEachIndexed { i, g ->
            val kind = if (g.overlapping) "hand" else if (g.cards.size == 1) "card" else "group"
            Text(
                "${kind.replaceFirstChar { it.uppercase() }} ${i + 1} · ${g.cards.size}:  " +
                    g.cards.sortedBy { it.quad.center.x }.joinToString("  ") { it.card.display },
                fontSize = 17.sp, fontWeight = FontWeight.Bold,
            )
        }
        scene.backs.forEach { b ->
            val what = when (b.layout) {
                BackLayout.SINGLE_OR_PILE -> "Face-down card or pile (e.g. draw deck)"
                BackLayout.SPREAD -> "Face-down spread · at least ${b.minCount} cards"
            }
            Text(what + if (b.fromTraining) "" else "  (guess; train backs)", color = BackColor, fontSize = 15.sp)
        }
        if (scene.unknown.isNotEmpty()) Text("${scene.unknown.size} card-like shape(s) not recognized", color = Color.Gray, fontSize = 12.sp)
        val footer = buildList {
            add("${scene.elapsedMs} ms")
            scene.cardHeightPx?.let { add("card ≈ ${it.toInt()} px tall") }
            if (backSamples == 0) add("tip: train your deck's back with the Back chip")
        }
        Text(footer.joinToString(" · "), color = Color.Gray, fontSize = 12.sp)
    }
}

/** Draws detections over an image shown with FIT_CENTER/ContentScale.Fit (frame size from the scene). */
@Composable
fun SceneOverlay(scene: TableScene, modifier: Modifier = Modifier) {
    val tm = rememberTextMeasurer()
    Canvas(modifier) {
        val sc = minOf(size.width / scene.frameWidth, size.height / scene.frameHeight)
        val ox = (size.width - scene.frameWidth * sc) / 2
        val oy = (size.height - scene.frameHeight * sc) / 2
        fun map(p: Pt) = Offset(ox + p.x * sc, oy + p.y * sc)
        for (g in scene.groups) {
            if (g.cards.size > 1) {
                val tl = map(Pt(g.box.left, g.box.top)); val br = map(Pt(g.box.right, g.box.bottom))
                drawRect(Color.White.copy(alpha = 0.6f), tl, androidx.compose.ui.geometry.Size(br.x - tl.x, br.y - tl.y),
                    style = Stroke(2f, pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 8f))))
            }
            for (c in g.cards) {
                val color = if (c.method == ReadMethod.FULL_CARD) FullCardColor else CornerColor
                drawQuad(c.quad, ::map, color)
                label(tm, c.card.display, map(c.quad.p0), color)
            }
        }
        for (b in scene.backs) {
            val q = b.quad ?: Quad(Pt(b.box.left, b.box.top), Pt(b.box.right, b.box.top), Pt(b.box.right, b.box.bottom), Pt(b.box.left, b.box.bottom))
            drawQuad(q, ::map, BackColor)
            label(tm, if (b.layout == BackLayout.SPREAD) "backs ≥${b.minCount}" else "back", map(q.center), BackColor)
        }
    }
}

private fun DrawScope.drawQuad(q: Quad, map: (Pt) -> Offset, color: Color) {
    val pts = q.points.map(map)
    val path = Path().apply { moveTo(pts[0].x, pts[0].y); pts.drop(1).forEach { lineTo(it.x, it.y) }; close() }
    drawPath(path, color, style = Stroke(4f))
}

private fun DrawScope.label(tm: TextMeasurer, text: String, at: Offset, color: Color) {
    val layout = tm.measure(text, TextStyle(color = Color.Black, fontSize = 15.sp, fontWeight = FontWeight.Bold))
    val tl = Offset(at.x.coerceIn(0f, size.width - layout.size.width), (at.y - layout.size.height).coerceIn(0f, size.height - layout.size.height))
    drawRect(color, tl, androidx.compose.ui.geometry.Size(layout.size.width.toFloat() + 8f, layout.size.height.toFloat()))
    drawText(layout, topLeft = tl + Offset(4f, 0f))
}

/**
 * Camera for the hand reader: higher analysis resolution than the trainer (far cards need pixels), a
 * full-resolution [ImageCapture] for snapshots, and a zoom slider. Preview, analysis and capture are all 4:3
 * so the overlay geometry matches.
 */
@Composable
private fun TableCamera(analyzer: ImageAnalysis.Analyzer, torch: Boolean, onCapture: (ImageCapture) -> Unit, modifier: Modifier) {
    val context = LocalContext.current
    val owner = LocalLifecycleOwner.current
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }
    var camera by remember { mutableStateOf<Camera?>(null) }
    var zoom by remember { mutableFloatStateOf(0f) }

    DisposableEffect(owner) {
        val executor = Executors.newSingleThreadExecutor()
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        future.addListener({
            val p = future.get().also { provider = it }
            fun selector(w: Int, h: Int) = ResolutionSelector.Builder()
                .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                .setResolutionStrategy(ResolutionStrategy(Size(w, h), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER))
                .build()
            val preview = Preview.Builder().setResolutionSelector(selector(1440, 1080)).build()
                .also { it.surfaceProvider = previewView.surfaceProvider }
            val analysis = ImageAnalysis.Builder()
                .setResolutionSelector(selector(1920, 1440))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build().also { it.setAnalyzer(executor, analyzer) }
            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                        .setResolutionStrategy(ResolutionStrategy.HIGHEST_AVAILABLE_STRATEGY)
                        .build()
                ).build()
            p.unbindAll()
            camera = p.bindToLifecycle(owner, CameraSelector.DEFAULT_BACK_CAMERA, preview, analysis, capture)
            onCapture(capture)
        }, ContextCompat.getMainExecutor(context))
        onDispose {
            provider?.unbindAll()
            executor.shutdown()
        }
    }
    LaunchedEffect(camera, torch) { camera?.cameraControl?.enableTorch(torch) }
    LaunchedEffect(camera, zoom) { camera?.cameraControl?.setLinearZoom(zoom) }

    Box(modifier) {
        AndroidView({ previewView }, Modifier.fillMaxSize())
        Row(
            Modifier.align(Alignment.BottomCenter).fillMaxWidth().padding(horizontal = 16.dp)
                .background(Color(0x66000000), MaterialTheme.shapes.small).padding(horizontal = 12.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text("Zoom", color = Color.White, fontSize = 13.sp)
            Slider(zoom, { zoom = it }, Modifier.padding(start = 8.dp))
        }
    }
}
