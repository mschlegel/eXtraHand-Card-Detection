package com.extrahand.cardtrainer

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.safeDrawingPadding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.AssistChip
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.extrahand.cards.android.CardFrameAnalyzer
import com.extrahand.cards.core.Card
import com.extrahand.cards.core.Rank
import com.extrahand.cards.core.Recognition
import com.extrahand.cards.core.Suit
import kotlinx.coroutines.delay

private val Green = Color(0xFF2E7D32)
private val Red = Color(0xFFC62828)

@Composable
fun TrainerScreen(vm: TrainerViewModel) {
    val s by vm.state.collectAsState()
    var torch by remember { mutableStateOf(false) }
    val analyzer = remember {
        CardFrameAnalyzer(recognizer = { null }, guide = GUIDE, minIntervalMs = 300, onFrame = vm::onFrame)
    }

    LaunchedEffect(s.message) { if (s.message != null) { delay(3500); vm.clearMessage() } }

    Column(Modifier.fillMaxSize().safeDrawingPadding()) {
        TopBar(s, torch, onTorch = { torch = !torch }, vm)
        Box(Modifier.weight(1f).fillMaxWidth().background(Color.Black)) {
            CameraPreview(analyzer, torch, Modifier.fillMaxSize())
            s.captured?.let { cap ->
                Image(
                    cap.image.asImageBitmap(), "captured card",
                    Modifier.fillMaxSize().background(Color.Black).padding(24.dp),
                    contentScale = ContentScale.Fit,
                )
            }
            if (s.captured == null) LiveBadge(s, Modifier.align(Alignment.TopCenter).padding(8.dp))
            s.message?.let {
                Text(
                    it, color = Color.White,
                    modifier = Modifier.align(Alignment.BottomCenter).padding(8.dp)
                        .background(Color(0xCC000000), MaterialTheme.shapes.small).padding(horizontal = 12.dp, vertical = 6.dp),
                )
            }
            if (s.loading) CircularProgressIndicator(Modifier.align(Alignment.Center))
        }
        BottomPanel(s, vm)
    }

    if (s.showSettings) SettingsDialog(s.settings, onDismiss = { vm.showSettings(false) }, onSave = vm::saveSettings)
}

@Composable
private fun TopBar(s: UiState, torch: Boolean, onTorch: () -> Unit, vm: TrainerViewModel) {
    Column(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 6.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text("Card Trainer", fontWeight = FontWeight.Bold, fontSize = 18.sp, modifier = Modifier.weight(1f))
            TextButton(onClick = onTorch) { Text(if (torch) "Light off" else "Light") }
            TextButton(onClick = vm::undoLast) { Text("Undo") }
            TextButton(onClick = { vm.sync() }, enabled = !s.syncing) { Text(if (s.syncing) "Syncing" else "Sync") }
            TextButton(onClick = { vm.showSettings(true) }) { Text("⚙") }
        }
        val acc = s.recentAccuracy?.let { "%.0f%%".format(it * 100) } ?: "–"
        Text(
            "${s.sampleCount} samples · ${s.labelsCovered}/52 cards · ${s.pendingCount} unsynced · " +
                "last $ACCURACY_WINDOW: $acc · v${s.datasetVersion}",
            fontSize = 12.sp, color = Color.LightGray,
        )
    }
}

@Composable
private fun LiveBadge(s: UiState, modifier: Modifier) {
    val r = s.live
    val text = when {
        s.sampleCount == 0 -> "No training data yet – capture a card and label it"
        r == null || r.candidates.isEmpty() -> "Hold a card inside the box"
        r.card != null -> "Sees ${r.card!!.display}  ${pct(r.confidence)}"
        else -> "Unsure (maybe ${r.candidates.first().card.display})"
    }
    Text(
        text, color = Color.White, fontSize = 16.sp,
        modifier = modifier.background(Color(0xAA000000), MaterialTheme.shapes.small).padding(horizontal = 12.dp, vertical = 6.dp),
    )
}

@Composable
private fun BottomPanel(s: UiState, vm: TrainerViewModel) {
    val cap = s.captured
    Column(Modifier.fillMaxWidth().padding(12.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
        when {
            cap == null -> Button(
                onClick = vm::capture, enabled = !s.loading,
                modifier = Modifier.fillMaxWidth().height(64.dp),
            ) { Text("Capture", fontSize = 20.sp) }

            s.picking -> CardPicker(
                initial = cap.guess,
                onPick = vm::submit,
                onBack = { if (cap.recognition.candidates.isEmpty()) vm.retake() else vm.showPicker(false) },
            )

            else -> GuessPanel(cap, vm)
        }
    }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun GuessPanel(cap: Captured, vm: TrainerViewModel) {
    val guess = cap.guess
    Text(
        when {
            guess == null -> "No guess"
            cap.guessConfident -> "Is it ${guess.display}?  (${pct(cap.recognition.confidence)})"
            else -> "Low confidence – is it ${guess.display}?"
        },
        fontSize = 22.sp, fontWeight = FontWeight.Bold,
    )
    Alternatives(cap.recognition, onPick = vm::submit)
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        Button(
            onClick = { guess?.let(vm::submit) }, enabled = guess != null,
            colors = ButtonDefaults.buttonColors(containerColor = Green),
            modifier = Modifier.weight(1f).height(56.dp),
        ) { Text("✓ Correct", fontSize = 18.sp) }
        Button(
            onClick = { vm.showPicker(true) },
            colors = ButtonDefaults.buttonColors(containerColor = Red),
            modifier = Modifier.weight(1f).height(56.dp),
        ) { Text("✗ Wrong", fontSize = 18.sp) }
    }
    OutlinedButton(onClick = vm::retake, modifier = Modifier.fillMaxWidth()) { Text("Retake (don't save)") }
}

@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun Alternatives(r: Recognition, onPick: (Card) -> Unit) {
    val alts = r.candidates.drop(1)
    if (alts.isEmpty()) return
    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Text("Or:", Modifier.align(Alignment.CenterVertically))
        alts.forEach { c -> AssistChip(onClick = { onPick(c.card) }, label = { Text("${c.card.display} ${pct(c.probability)}") }) }
    }
}

/** Two-tap picker: rank + suit (or Joker), then Save. Pre-selects the guess since one half is often right. */
@OptIn(ExperimentalLayoutApi::class)
@Composable
private fun CardPicker(initial: Card?, onPick: (Card) -> Unit, onBack: () -> Unit) {
    var rank by remember { mutableStateOf(initial?.rank) }
    var suit by remember { mutableStateOf(initial?.suit) }
    Text("Which card is it?", fontWeight = FontWeight.Bold)
    FlowRow(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
        Rank.entries.forEach { r ->
            FilterChip(selected = rank == r, onClick = { rank = r }, label = { Text(r.display, fontSize = 16.sp) })
        }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(6.dp)) {
        Suit.entries.forEach { su ->
            FilterChip(
                selected = suit == su, onClick = { suit = su },
                label = { Text(su.symbol, fontSize = 22.sp, color = if (su.isRed) Color(0xFFEF5350) else Color.Unspecified) },
            )
        }
        Spacer(Modifier.width(8.dp))
        AssistChip(onClick = { onPick(Card.JOKER) }, label = { Text("Joker") })
    }
    val chosen = if (rank != null && suit != null) Card(rank, suit) else null
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        OutlinedButton(onClick = onBack, modifier = Modifier.weight(1f)) { Text("Back") }
        Button(
            onClick = { chosen?.let(onPick) }, enabled = chosen != null,
            modifier = Modifier.weight(2f),
        ) { Text(chosen?.let { "Save as ${it.display}" } ?: "Pick rank + suit") }
    }
}

@Composable
private fun SettingsDialog(initial: TrainerSettings, onDismiss: () -> Unit, onSave: (TrainerSettings) -> Unit) {
    var owner by remember { mutableStateOf(initial.owner) }
    var repo by remember { mutableStateOf(initial.repo) }
    var branch by remember { mutableStateOf(initial.branch) }
    var path by remember { mutableStateOf(initial.path) }
    var token by remember { mutableStateOf(initial.token) }
    var every by remember { mutableStateOf(initial.autoSyncEvery.toString()) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("GitHub sync") },
        text = {
            Column(Modifier.verticalScroll(rememberScrollState()), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                OutlinedTextField(owner, { owner = it }, label = { Text("Owner") }, singleLine = true)
                OutlinedTextField(repo, { repo = it }, label = { Text("Repository") }, singleLine = true)
                OutlinedTextField(branch, { branch = it }, label = { Text("Branch") }, singleLine = true)
                OutlinedTextField(path, { path = it }, label = { Text("Dataset folder") }, singleLine = true)
                OutlinedTextField(
                    token, { token = it }, label = { Text("Token (Contents: read & write)") }, singleLine = true,
                    visualTransformation = PasswordVisualTransformation(),
                )
                OutlinedTextField(
                    every, { every = it.filter(Char::isDigit) }, label = { Text("Auto-sync every N changes (0 = off)") },
                    singleLine = true, keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
                )
                Text("Trained data is published to <repo>/<folder>/ on every sync.", fontSize = 12.sp, color = Color.Gray)
            }
        },
        confirmButton = {
            TextButton(onClick = {
                onSave(TrainerSettings(owner.trim(), repo.trim(), branch.trim(), path.trim(), token.trim(), every.toIntOrNull() ?: 0))
            }) { Text("Save") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

private fun pct(p: Float) = "%.0f%%".format(p * 100)

