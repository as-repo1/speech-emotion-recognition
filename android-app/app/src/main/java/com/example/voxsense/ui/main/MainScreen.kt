package com.example.voxsense.ui.main

import android.Manifest
import android.content.pm.PackageManager
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.*
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.voxsense.data.DefaultDataRepository
import com.example.voxsense.data.EmotionPrediction
import com.example.voxsense.data.ModelManager
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Close
import androidx.compose.material.icons.filled.Delete
import androidx.compose.foundation.BorderStroke
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.PI
import kotlin.math.sin

// --- NORD THEME COLOR PALETTE DEFINITION ---
val Nord0 = Color(0xFF2E3440) // Polar Night (Darkest)
val Nord1 = Color(0xFF3B4252) // Polar Night (Card)
val Nord2 = Color(0xFF434C5E) // Polar Night (Borders)
val Nord3 = Color(0xFF4C566A) // Polar Night (Lighter Gray / Unselected)
val Nord4 = Color(0xFFD8DEE9) // Snow Storm (Text)
val Nord5 = Color(0xFFE5E9F0) // Snow Storm (Titles)
val Nord6 = Color(0xFFECEFF4) // Snow Storm (Bright Text)
val Nord7 = Color(0xFF8FBCBB) // Frost (Teal / Secondary Highlight)
val Nord8 = Color(0xFF88C0D0) // Frost (Blue / Primary Highlight)
val Nord9 = Color(0xFF81A1C1) // Frost (Classic Blue)
val Nord10 = Color(0xFF5E81AC) // Frost (Dark Blue)

// Aurora Emotion Colors
val Nord11 = Color(0xFFBF616A) // Aurora Red (Angry)
val Nord12 = Color(0xFFD08770) // Aurora Orange (Surprised)
val Nord13 = Color(0xFFEBCB8B) // Aurora Yellow (Happy)
val Nord14 = Color(0xFFA3BE8C) // Aurora Green (Disgust)
val Nord15 = Color(0xFFB48EAD) // Aurora Purple (Fearful)

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MainScreen(
    onItemClick: (androidx.navigation3.runtime.NavKey) -> Unit,
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current
    val viewModel: MainScreenViewModel = viewModel {
        MainScreenViewModel(DefaultDataRepository(context.applicationContext))
    }

    val state by viewModel.uiState.collectAsStateWithLifecycle()
    val isRecording by viewModel.isRecording.collectAsStateWithLifecycle()
    val isProcessing by viewModel.isProcessing.collectAsStateWithLifecycle()
    val amplitude by viewModel.currentAmplitude.collectAsStateWithLifecycle()
    val duration by viewModel.recordingDuration.collectAsStateWithLifecycle()
    val modelStates by viewModel.modelStates.collectAsStateWithLifecycle()
    val downloadError by viewModel.downloadError.collectAsStateWithLifecycle()

    var hasMicPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED
        )
    }

    val permissionLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        hasMicPermission = isGranted
    }

    val Nord0 = MaterialTheme.colorScheme.background
    val Nord1 = MaterialTheme.colorScheme.surfaceVariant
    val Nord2 = MaterialTheme.colorScheme.outline
    val Nord3 = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
    val Nord4 = MaterialTheme.colorScheme.onSurfaceVariant
    val Nord5 = MaterialTheme.colorScheme.onSurface
    val Nord6 = MaterialTheme.colorScheme.onSurface
    val Nord7 = MaterialTheme.colorScheme.secondary
    val Nord8 = MaterialTheme.colorScheme.primary
    val Nord9 = MaterialTheme.colorScheme.primary.copy(alpha = 0.8f)
    val Nord10 = MaterialTheme.colorScheme.primaryContainer
    val Nord11 = MaterialTheme.colorScheme.error

    // Nord Background Gradient (Polar Night tone)
    val backgroundGradient = Brush.verticalGradient(
        colors = listOf(Nord0, MaterialTheme.colorScheme.surfaceVariant)
    )

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(backgroundGradient)
            .padding(16.dp)
    ) {
        when (state) {
            is MainScreenUiState.Success -> {
                val successState = state as MainScreenUiState.Success
                MainScreenContent(
                    latestPrediction = successState.latestPrediction,
                    history = successState.history,
                    isRecording = isRecording,
                    isProcessing = isProcessing,
                    amplitude = amplitude,
                    duration = duration,
                    hasPermission = hasMicPermission,
                    enabledEmotions = successState.enabledEmotions,
                    usePreEmphasis = successState.usePreEmphasis,
                    useNoiseGate = successState.useNoiseGate,
                    recordingLimitSeconds = successState.recordingLimitSeconds,
                    modelStates = modelStates,
                    downloadError = downloadError,
                    onSelectModel = { viewModel.selectModel(it) },
                    onDeleteModel = { viewModel.deleteModel(it) },
                    onDownloadModel = { viewModel.downloadModel(it) },
                    onClearDownloadError = { viewModel.clearDownloadError() },
                    onRequestPermission = {
                        permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
                    },
                    onRecordToggle = {
                        if (isRecording) {
                            viewModel.stopRecording()
                        } else {
                            viewModel.startRecording()
                        }
                    },
                    onClearHistory = {
                        viewModel.clearHistory()
                    },
                    onToggleEmotion = { emotion ->
                        viewModel.toggleEmotion(emotion)
                    },
                    onTogglePreEmphasis = {
                        viewModel.togglePreEmphasis()
                    },
                    onToggleNoiseGate = {
                        viewModel.toggleNoiseGate()
                    },
                    onSetRecordingLimit = { seconds ->
                        viewModel.setRecordingLimit(seconds)
                    }
                )
            }
            is MainScreenUiState.Error -> {
                Box(modifier = Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
                    Text(
                        text = "Error: ${(state as MainScreenUiState.Error).throwable.message}",
                        color = Nord11,
                        fontSize = 18.sp
                    )
                }
            }
        }
    }
}

@Composable
fun MainScreenContent(
    latestPrediction: EmotionPrediction?,
    history: List<EmotionPrediction>,
    isRecording: Boolean,
    isProcessing: Boolean,
    amplitude: Float,
    duration: Long,
    hasPermission: Boolean,
    enabledEmotions: Set<String>,
    usePreEmphasis: Boolean,
    useNoiseGate: Boolean,
    recordingLimitSeconds: Int?,
    modelStates: List<ModelUIState>,
    downloadError: String?,
    onSelectModel: (String) -> Unit,
    onDeleteModel: (String) -> Unit,
    onDownloadModel: (String) -> Unit,
    onClearDownloadError: () -> Unit,
    onRequestPermission: () -> Unit,
    onRecordToggle: () -> Unit,
    onClearHistory: () -> Unit,
    onToggleEmotion: (String) -> Unit,
    onTogglePreEmphasis: () -> Unit,
    onToggleNoiseGate: () -> Unit,
    onSetRecordingLimit: (Int?) -> Unit
) {
    val Nord0 = MaterialTheme.colorScheme.background
    val Nord1 = MaterialTheme.colorScheme.surfaceVariant
    val Nord2 = MaterialTheme.colorScheme.outline
    val Nord3 = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
    val Nord4 = MaterialTheme.colorScheme.onSurfaceVariant
    val Nord5 = MaterialTheme.colorScheme.onSurface
    val Nord6 = MaterialTheme.colorScheme.onSurface
    val Nord7 = MaterialTheme.colorScheme.secondary
    val Nord8 = MaterialTheme.colorScheme.primary
    val Nord9 = MaterialTheme.colorScheme.primary.copy(alpha = 0.8f)
    val Nord10 = MaterialTheme.colorScheme.primaryContainer
    val Nord11 = MaterialTheme.colorScheme.error

    var settingsExpanded by remember { mutableStateOf(false) }

    if (isProcessing) {
        Dialog(
            onDismissRequest = {},
            properties = DialogProperties(dismissOnBackPress = false, dismissOnClickOutside = false)
        ) {
            Box(
                modifier = Modifier
                    .size(175.dp)
                    .background(MaterialTheme.colorScheme.surfaceVariant, RoundedCornerShape(16.dp))
                    .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(16.dp))
                    .padding(16.dp),
                contentAlignment = Alignment.Center
            ) {
                Column(
                    horizontalAlignment = Alignment.CenterHorizontally,
                    verticalArrangement = Arrangement.Center
                ) {
                    CircularProgressIndicator(
                        color = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.size(48.dp)
                    )
                    Spacer(modifier = Modifier.height(16.dp))
                    Text(
                        text = "Analyzing Speech...",
                        color = MaterialTheme.colorScheme.onSurface,
                        fontSize = 14.sp,
                        fontWeight = FontWeight.Bold,
                        textAlign = TextAlign.Center
                    )
                    Spacer(modifier = Modifier.height(4.dp))
                    Text(
                        text = "Extracting acoustic features",
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontSize = 10.sp,
                        textAlign = TextAlign.Center
                    )
                }
            }
        }
    }

    LazyColumn(
        modifier = Modifier.fillMaxSize(),
        verticalArrangement = Arrangement.spacedBy(16.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // App Header (Minimalist Nord Branding)
        item {
            Spacer(modifier = Modifier.height(8.dp))
            Text(
                text = "VOXSENSE",
                fontSize = 26.sp,
                fontWeight = FontWeight.Bold,
                color = Nord6,
                letterSpacing = 4.sp
            )
            Text(
                text = "Offline Speech Emotion Classifier",
                color = Nord4.copy(alpha = 0.7f),
                fontSize = 12.sp,
                fontWeight = FontWeight.Normal,
                modifier = Modifier.padding(top = 4.dp)
            )
        }

        // Expandable Settings Row
        item {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .clickable { settingsExpanded = !settingsExpanded }
                    .padding(vertical = 8.dp, horizontal = 4.dp),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "Customization Options",
                    color = Nord5,
                    fontSize = 15.sp,
                    fontWeight = FontWeight.Bold
                )
                Text(
                    text = if (settingsExpanded) "Collapse ▲" else "Expand Settings ▼",
                    color = Nord8,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // Settings Expansion Card
        if (settingsExpanded) {
            item {
                NordCard(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(14.dp)
                    ) {
                        // Signal Processing Settings
                        Text(
                            text = "Acoustic Processing",
                            color = Nord5,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(16.dp)
                        ) {
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.weight(1f)
                            ) {
                                Switch(
                                    checked = usePreEmphasis,
                                    onCheckedChange = { onTogglePreEmphasis() },
                                    colors = SwitchDefaults.colors(
                                        checkedThumbColor = Nord6,
                                        checkedTrackColor = Nord8,
                                        uncheckedThumbColor = Nord4,
                                        uncheckedTrackColor = Nord2
                                    )
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Column {
                                    Text("Pre-emphasis", color = Nord4, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                    Text("High-freq boost", color = Nord4.copy(alpha = 0.6f), fontSize = 10.sp)
                                }
                            }
                            Row(
                                verticalAlignment = Alignment.CenterVertically,
                                modifier = Modifier.weight(1f)
                            ) {
                                Switch(
                                    checked = useNoiseGate,
                                    onCheckedChange = { onToggleNoiseGate() },
                                    colors = SwitchDefaults.colors(
                                        checkedThumbColor = Nord6,
                                        checkedTrackColor = Nord8,
                                        uncheckedThumbColor = Nord4,
                                        uncheckedTrackColor = Nord2
                                    )
                                )
                                Spacer(modifier = Modifier.width(8.dp))
                                Column {
                                    Text("Noise Gate", color = Nord4, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                                    Text("Silence gating", color = Nord4.copy(alpha = 0.6f), fontSize = 10.sp)
                                }
                            }
                        }

                        HorizontalDivider(color = Nord2, thickness = 1.dp)

                        // Recording Timer Limit Selection
                        Text(
                            text = "Recording Duration",
                            color = Nord5,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            NordChoiceChip(
                                text = "Manual Stop",
                                selected = recordingLimitSeconds == null,
                                onClick = { onSetRecordingLimit(null) },
                                modifier = Modifier.weight(1f)
                            )
                            NordChoiceChip(
                                text = "3s",
                                selected = recordingLimitSeconds == 3,
                                onClick = { onSetRecordingLimit(3) },
                                modifier = Modifier.weight(0.7f)
                            )
                            NordChoiceChip(
                                text = "5s",
                                selected = recordingLimitSeconds == 5,
                                onClick = { onSetRecordingLimit(5) },
                                modifier = Modifier.weight(0.7f)
                            )
                            NordChoiceChip(
                                text = "10s",
                                selected = recordingLimitSeconds == 10,
                                onClick = { onSetRecordingLimit(10) },
                                modifier = Modifier.weight(0.7f)
                            )
                        }

                        HorizontalDivider(color = Nord2, thickness = 1.dp)

                        // Emotions Multi-select Filters
                        Text(
                            text = "Tracked Emotions",
                            color = Nord5,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                        
                        Column(
                            verticalArrangement = Arrangement.spacedBy(8.dp),
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                NordEmotionToggleChip(
                                    emotion = "neutral",
                                    selected = "neutral" in enabledEmotions,
                                    onClick = { onToggleEmotion("neutral") },
                                    modifier = Modifier.weight(1f)
                                )
                                NordEmotionToggleChip(
                                    emotion = "calm",
                                    selected = "calm" in enabledEmotions,
                                    onClick = { onToggleEmotion("calm") },
                                    modifier = Modifier.weight(1f)
                                )
                                NordEmotionToggleChip(
                                    emotion = "happy",
                                    selected = "happy" in enabledEmotions,
                                    onClick = { onToggleEmotion("happy") },
                                    modifier = Modifier.weight(1f)
                                )
                            }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                NordEmotionToggleChip(
                                    emotion = "sad",
                                    selected = "sad" in enabledEmotions,
                                    onClick = { onToggleEmotion("sad") },
                                    modifier = Modifier.weight(1f)
                                )
                                NordEmotionToggleChip(
                                    emotion = "angry",
                                    selected = "angry" in enabledEmotions,
                                    onClick = { onToggleEmotion("angry") },
                                    modifier = Modifier.weight(1f)
                                )
                                NordEmotionToggleChip(
                                    emotion = "fearful",
                                    selected = "fearful" in enabledEmotions,
                                    onClick = { onToggleEmotion("fearful") },
                                    modifier = Modifier.weight(1f)
                                )
                            }
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                NordEmotionToggleChip(
                                    emotion = "disgust",
                                    selected = "disgust" in enabledEmotions,
                                    onClick = { onToggleEmotion("disgust") },
                                    modifier = Modifier.weight(1f)
                                )
                                NordEmotionToggleChip(
                                    emotion = "surprised",
                                    selected = "surprised" in enabledEmotions,
                                    onClick = { onToggleEmotion("surprised") },
                                    modifier = Modifier.weight(1f)
                                )
                                Spacer(modifier = Modifier.weight(1f)) // Padding
                            }
                        }
                    }
                }
            }

            // Open Source Model Downloader Card
            item {
                Spacer(modifier = Modifier.height(8.dp))
                NordCard(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        modifier = Modifier.fillMaxWidth(),
                        verticalArrangement = Arrangement.spacedBy(12.dp)
                    ) {
                        Text(
                            text = "Model Management",
                            color = Nord5,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold
                        )
                        Text(
                            text = "Download advanced neural networks directly from open sources to local storage.",
                            color = Nord4.copy(alpha = 0.6f),
                            fontSize = 11.sp
                        )

                        if (downloadError != null) {
                            Card(
                                colors = CardDefaults.cardColors(containerColor = Nord11.copy(alpha = 0.15f)),
                                border = BorderStroke(1.dp, Nord11),
                                modifier = Modifier.fillMaxWidth()
                            ) {
                                Row(
                                    modifier = Modifier.padding(10.dp).fillMaxWidth(),
                                    verticalAlignment = Alignment.CenterVertically,
                                    horizontalArrangement = Arrangement.SpaceBetween
                                ) {
                                    Text(
                                        text = downloadError,
                                        color = Nord11,
                                        fontSize = 11.sp,
                                        modifier = Modifier.weight(1f)
                                    )
                                    IconButton(
                                        onClick = onClearDownloadError,
                                        modifier = Modifier.size(24.dp)
                                    ) {
                                        Icon(
                                            imageVector = Icons.Default.Close,
                                            contentDescription = "Clear Error",
                                            tint = Nord11,
                                            modifier = Modifier.size(16.dp)
                                        )
                                    }
                                }
                            }
                        }

                        modelStates.forEach { state ->
                            val model = state.model
                            Row(
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .border(1.dp, if (state.isActive) Nord8 else Nord2, RoundedCornerShape(8.dp))
                                    .background(if (state.isActive) Nord8.copy(alpha = 0.05f) else Color.Transparent, RoundedCornerShape(8.dp))
                                    .padding(10.dp),
                                verticalAlignment = Alignment.CenterVertically,
                                horizontalArrangement = Arrangement.SpaceBetween
                            ) {
                                Column(modifier = Modifier.weight(1f)) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Text(
                                            text = model.name,
                                            color = if (state.isActive) Nord8 else Nord4,
                                            fontSize = 12.sp,
                                            fontWeight = FontWeight.Bold
                                        )
                                        Spacer(modifier = Modifier.width(6.dp))
                                        Text(
                                            text = model.size,
                                            color = Nord7,
                                            fontSize = 9.sp,
                                            fontWeight = FontWeight.Bold,
                                            modifier = Modifier
                                                .background(Nord7.copy(alpha = 0.1f), RoundedCornerShape(3.dp))
                                                .padding(horizontal = 4.dp, vertical = 2.dp)
                                        )
                                    }
                                    Spacer(modifier = Modifier.height(2.dp))
                                    Text(
                                        text = model.description,
                                        color = Nord4.copy(alpha = 0.6f),
                                        fontSize = 10.sp
                                    )
                                }

                                Spacer(modifier = Modifier.width(8.dp))

                                when {
                                    state.isDownloading -> {
                                        Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                            CircularProgressIndicator(
                                                progress = (state.downloadProgress ?: 0) / 100f,
                                                color = Nord8,
                                                modifier = Modifier.size(20.dp),
                                                strokeWidth = 2.dp
                                            )
                                            Text(
                                                text = "${state.downloadProgress ?: 0}%",
                                                color = Nord4,
                                                fontSize = 9.sp,
                                                modifier = Modifier.padding(top = 2.dp)
                                            )
                                        }
                                    }
                                    state.isDownloaded -> {
                                        Row(verticalAlignment = Alignment.CenterVertically) {
                                            if (state.isActive) {
                                                Text(
                                                    text = "Active",
                                                    color = Nord14,
                                                    fontSize = 10.sp,
                                                    fontWeight = FontWeight.Bold,
                                                    modifier = Modifier
                                                        .border(1.dp, Nord14, RoundedCornerShape(4.dp))
                                                        .padding(horizontal = 6.dp, vertical = 2.dp)
                                                )
                                            } else {
                                                Button(
                                                    onClick = { onSelectModel(model.id) },
                                                    colors = ButtonDefaults.buttonColors(containerColor = Nord8),
                                                    contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                                                    modifier = Modifier.height(28.dp)
                                                ) {
                                                    Text("Activate", color = Nord0, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                                                }
                                                if (model.id != "cnn") {
                                                    Spacer(modifier = Modifier.width(4.dp))
                                                    IconButton(
                                                        onClick = { onDeleteModel(model.id) },
                                                        modifier = Modifier.size(28.dp)
                                                    ) {
                                                        Icon(
                                                            imageVector = Icons.Default.Delete,
                                                            contentDescription = "Delete model",
                                                            tint = Nord11,
                                                            modifier = Modifier.size(16.dp)
                                                        )
                                                    }
                                                }
                                            }
                                        }
                                    }
                                    else -> {
                                        Button(
                                            onClick = { onDownloadModel(model.id) },
                                            colors = ButtonDefaults.buttonColors(containerColor = Nord3),
                                            contentPadding = PaddingValues(horizontal = 8.dp, vertical = 2.dp),
                                            modifier = Modifier.height(28.dp)
                                        ) {
                                            Text("Download", color = Nord6, fontSize = 10.sp, fontWeight = FontWeight.Bold)
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Minimalist Sine Wave Visualizer Card
        item {
            NordCard(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(130.dp)
            ) {
                Box(
                    modifier = Modifier.fillMaxSize(),
                    contentAlignment = Alignment.Center
                ) {
                    NordWaveformVisualizer(
                        isActive = isRecording,
                        amplitude = amplitude
                    )

                    if (isRecording) {
                        val timeStr = formatDuration(duration)
                        val limitStr = if (recordingLimitSeconds != null) "/$recordingLimitSeconds" else ""
                        Text(
                            text = "$timeStr$limitStr",
                            color = Nord6,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            modifier = Modifier
                                .align(Alignment.BottomCenter)
                                .padding(bottom = 8.dp)
                                .background(Nord0.copy(alpha = 0.7f), CircleShape)
                                .padding(horizontal = 12.dp, vertical = 4.dp)
                        )
                    } else if (isProcessing) {
                        CircularProgressIndicator(
                            color = Nord8,
                            modifier = Modifier.size(32.dp)
                        )
                    } else {
                        Text(
                            text = "Tap Record to Classify",
                            color = Nord3,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }
            }
        }

        // Dynamic Record Button
        item {
            if (!hasPermission) {
                Button(
                    onClick = onRequestPermission,
                    colors = ButtonDefaults.buttonColors(containerColor = Nord11),
                    shape = RoundedCornerShape(24.dp)
                ) {
                    Text("Grant Microphone Access", color = Nord6, fontWeight = FontWeight.Bold)
                }
            } else {
                NordRecordButton(
                    isRecording = isRecording,
                    isProcessing = isProcessing,
                    onClick = onRecordToggle
                )
            }
        }

        // Analysis Results Card (shows filtered predictions)
        if (latestPrediction != null && !isRecording && !isProcessing) {
            item {
                Text(
                    text = "Acoustic Analysis",
                    color = Nord6,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Bold,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 4.dp),
                    textAlign = TextAlign.Start
                )
            }

            item {
                NordCard(modifier = Modifier.fillMaxWidth()) {
                    Column(
                        horizontalAlignment = Alignment.CenterHorizontally,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        val emotion = latestPrediction.primaryEmotion
                        val emoji = getEmotionEmoji(emotion)
                        val color = getEmotionColor(emotion)
                        val confPct = (latestPrediction.confidence * 100).toInt()

                        Text(
                            text = emoji,
                            fontSize = 56.sp,
                            modifier = Modifier.padding(bottom = 4.dp)
                        )

                        Text(
                            text = emotion.uppercase(Locale.ROOT),
                            color = color,
                            fontSize = 20.sp,
                            fontWeight = FontWeight.Bold
                        )

                        Text(
                            text = "Confidence Score: $confPct%",
                            color = Nord4.copy(alpha = 0.7f),
                            fontSize = 13.sp,
                            modifier = Modifier.padding(top = 2.dp, bottom = 14.dp)
                        )

                        HorizontalDivider(color = Nord2, thickness = 1.dp)
                        Spacer(modifier = Modifier.height(14.dp))

                        // Render horizontal charts ONLY for emotions selected in enabledEmotions
                        val filteredList = latestPrediction.allEmotions.entries
                            .filter { it.key in enabledEmotions }
                            .sortedByDescending { it.value }

                        if (filteredList.isNotEmpty()) {
                            filteredList.forEach { (emo, conf) ->
                                NordEmotionBar(
                                    label = emo,
                                    value = conf,
                                    color = getEmotionColor(emo)
                                )
                                Spacer(modifier = Modifier.height(10.dp))
                            }
                        } else {
                            Text(
                                text = "All tracked emotions filtered out.",
                                color = Nord3,
                                fontSize = 12.sp,
                                modifier = Modifier.padding(vertical = 8.dp)
                            )
                        }
                    }
                }
            }
        }

        // Recent Predictions History
        if (history.isNotEmpty()) {
            item {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "History Timeline",
                        color = Nord6,
                        fontSize = 16.sp,
                        fontWeight = FontWeight.Bold
                    )
                    TextButton(
                        onClick = onClearHistory,
                        colors = ButtonDefaults.textButtonColors(contentColor = Nord11)
                    ) {
                        Text("Clear All", fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }

            items(history) { item ->
                NordCard(
                    modifier = Modifier.fillMaxWidth(),
                    shape = RoundedCornerShape(12.dp)
                ) {
                    Row(
                        verticalAlignment = Alignment.CenterVertically,
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = getEmotionEmoji(item.primaryEmotion),
                            fontSize = 28.sp,
                            modifier = Modifier.padding(end = 12.dp)
                        )

                        Column(modifier = Modifier.weight(1f)) {
                            Text(
                                text = item.primaryEmotion.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString() },
                                color = Nord5,
                                fontWeight = FontWeight.Bold,
                                fontSize = 15.sp
                            )
                            Text(
                                text = formatTime(item.timestamp),
                                color = Nord3,
                                fontSize = 11.sp
                            )
                        }

                        Text(
                            text = "${(item.confidence * 100).toInt()}%",
                            color = getEmotionColor(item.primaryEmotion),
                            fontWeight = FontWeight.Bold,
                            fontSize = 15.sp
                        )
                    }
                }
            }
        }

        item {
            Spacer(modifier = Modifier.height(24.dp))
        }
    }
}

@Composable
fun NordCard(
    modifier: Modifier = Modifier,
    shape: RoundedCornerShape = RoundedCornerShape(14.dp),
    content: @Composable () -> Unit
) {
    val Nord1 = MaterialTheme.colorScheme.surfaceVariant
    val Nord2 = MaterialTheme.colorScheme.outline
    Box(
        modifier = modifier
            .clip(shape)
            .background(Nord1.copy(alpha = 0.85f))
            .border(1.dp, Nord2.copy(alpha = 0.7f), shape)
            .padding(14.dp)
    ) {
        content()
    }
}

@Composable
fun NordChoiceChip(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val Nord0 = MaterialTheme.colorScheme.background
    val Nord2 = MaterialTheme.colorScheme.surfaceVariant
    val Nord3 = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
    val Nord4 = MaterialTheme.colorScheme.onSurfaceVariant
    val Nord8 = MaterialTheme.colorScheme.primary
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(18.dp))
            .background(if (selected) Nord8 else Nord2.copy(alpha = 0.6f))
            .border(1.dp, if (selected) Nord8 else Nord3.copy(alpha = 0.5f), RoundedCornerShape(18.dp))
            .clickable { onClick() }
            .padding(vertical = 8.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = text,
            color = if (selected) Nord0 else Nord4,
            fontSize = 12.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun NordEmotionToggleChip(
    emotion: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier
) {
    val Nord2 = MaterialTheme.colorScheme.surfaceVariant
    val Nord4 = MaterialTheme.colorScheme.onSurfaceVariant
    val Nord6 = MaterialTheme.colorScheme.onSurface
    val color = getEmotionColor(emotion)
    val emoji = getEmotionEmoji(emotion)
    Box(
        modifier = modifier
            .clip(RoundedCornerShape(14.dp))
            .background(if (selected) color.copy(alpha = 0.15f) else Nord2.copy(alpha = 0.3f))
            .border(1.dp, if (selected) color else Nord2.copy(alpha = 0.6f), RoundedCornerShape(14.dp))
            .clickable { onClick() }
            .padding(vertical = 8.dp, horizontal = 6.dp),
        contentAlignment = Alignment.Center
    ) {
        Row(
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.Center
        ) {
            Text(text = emoji, modifier = Modifier.padding(end = 4.dp), fontSize = 13.sp)
            Text(
                text = emotion.replaceFirstChar { it.titlecase(Locale.ROOT) },
                color = if (selected) Nord6 else Nord4.copy(alpha = 0.5f),
                fontSize = 11.sp,
                fontWeight = if (selected) FontWeight.Bold else FontWeight.Medium
            )
        }
    }
}

@Composable
fun NordRecordButton(
    isRecording: Boolean,
    isProcessing: Boolean,
    onClick: () -> Unit
) {
    val Nord0 = MaterialTheme.colorScheme.background
    val Nord3 = MaterialTheme.colorScheme.onSurface.copy(alpha = 0.4f)
    val Nord6 = MaterialTheme.colorScheme.onSurface
    val Nord8 = MaterialTheme.colorScheme.primary
    val Nord11 = MaterialTheme.colorScheme.error

    val infiniteTransition = rememberInfiniteTransition(label = "pulse")
    val glowSize by if (isRecording) {
        infiniteTransition.animateFloat(
            initialValue = 0.dp.value,
            targetValue = 12.dp.value,
            animationSpec = infiniteRepeatable(
                animation = tween(900, easing = FastOutSlowInEasing),
                repeatMode = RepeatMode.Reverse
            ),
            label = "buttonGlow"
        )
    } else {
        remember { mutableStateOf(0.dp.value) }
    }

    val buttonColor = if (isRecording) Nord11 else Nord8
    val buttonText = if (isRecording) "Stop" else "Record"

    Button(
        onClick = onClick,
        enabled = !isProcessing,
        colors = ButtonDefaults.buttonColors(
            containerColor = buttonColor,
            disabledContainerColor = Nord3
        ),
        modifier = Modifier
            .size(width = 150.dp, height = 50.dp)
            .border(
                width = glowSize.dp,
                color = buttonColor.copy(alpha = 0.25f),
                shape = RoundedCornerShape(25.dp)
            ),
        shape = RoundedCornerShape(25.dp),
        elevation = ButtonDefaults.buttonElevation(defaultElevation = 2.dp)
    ) {
        Text(
            text = buttonText,
            color = if (isRecording) Nord6 else Nord0,
            fontSize = 15.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
fun NordEmotionBar(
    label: String,
    value: Float,
    color: Color
) {
    val Nord2 = MaterialTheme.colorScheme.surfaceVariant
    val Nord4 = MaterialTheme.colorScheme.onSurface

    Column(modifier = Modifier.fillMaxWidth()) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = label.replaceFirstChar { if (it.isLowerCase()) it.titlecase(Locale.ROOT) else it.toString() },
                color = Nord4,
                fontSize = 12.sp,
                fontWeight = FontWeight.SemiBold
            )
            Text(
                text = "${(value * 100).toInt()}%",
                color = color,
                fontSize = 12.sp,
                fontWeight = FontWeight.Bold
            )
        }
        Spacer(modifier = Modifier.height(4.dp))
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .height(6.dp)
                .clip(CircleShape)
                .background(Nord2)
        ) {
            val animatedWidth = animateFloatAsState(
                targetValue = value,
                animationSpec = tween(900, easing = FastOutSlowInEasing),
                label = "barWidth"
            )
            Box(
                modifier = Modifier
                    .fillMaxHeight()
                    .fillMaxWidth(animatedWidth.value)
                    .clip(CircleShape)
                    .background(color)
            )
        }
    }
}

@Composable
fun NordWaveformVisualizer(
    isActive: Boolean,
    amplitude: Float
) {
    val Nord8 = MaterialTheme.colorScheme.primary
    val Nord15 = MaterialTheme.colorScheme.secondary

    val infiniteTransition = rememberInfiniteTransition(label = "waveAnim")
    val phase by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = (2 * PI).toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(1400, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "phase"
    )

    val smoothAmplitude = animateFloatAsState(
        targetValue = if (isActive) amplitude else 0.04f,
        animationSpec = tween(120, easing = LinearEasing),
        label = "amplitude"
    )

    Canvas(modifier = Modifier.fillMaxSize()) {
        val width = size.width
        val height = size.height
        val centerY = height / 2

        val path1 = Path()
        val path2 = Path()

        val ampVal1 = smoothAmplitude.value * (height * 0.35f)
        val ampVal2 = smoothAmplitude.value * (height * 0.2f)

        for (x in 0..width.toInt() step 6) {
            val progress = x.toFloat() / width
            val windowScale = sin(progress * PI.toFloat())

            val y1 = centerY + sin(progress * 13f - phase) * ampVal1 * windowScale
            val y2 = centerY + sin(progress * 19f + phase * 1.3f) * ampVal2 * windowScale

            if (x == 0) {
                path1.moveTo(0f, y1)
                path2.moveTo(0f, y2)
            } else {
                path1.lineTo(x.toFloat(), y1)
                path2.lineTo(x.toFloat(), y2)
            }
        }

        drawPath(
            path = path1,
            color = Nord8.copy(alpha = 0.8f),
            style = Stroke(width = 2.dp.toPx())
        )
        drawPath(
            path = path2,
            color = Nord15.copy(alpha = 0.5f),
            style = Stroke(width = 1.5.dp.toPx())
        )
    }
}

private fun formatDuration(millis: Long): String {
    val seconds = (millis / 1000) % 60
    val minutes = (millis / 60000) % 60
    return String.format(Locale.getDefault(), "%02d:%02d", minutes, seconds)
}

private fun formatTime(timestamp: Long): String {
    val sdf = SimpleDateFormat("MMM d, h:mm a", Locale.getDefault())
    return sdf.format(Date(timestamp))
}

fun getEmotionEmoji(emotion: String): String = when (emotion.lowercase(Locale.ROOT)) {
    "neutral" -> "😐"
    "calm" -> "😌"
    "happy" -> "😄"
    "sad" -> "😢"
    "angry" -> "😡"
    "fearful" -> "😨"
    "disgust" -> "🤢"
    "surprised" -> "😲"
    else -> "❓"
}

fun getEmotionColor(emotion: String): Color = when (emotion.lowercase(Locale.ROOT)) {
    "neutral" -> Nord4      // Snow Storm Gray
    "calm" -> Nord7         // Frost Teal
    "happy" -> Nord13       // Aurora Yellow
    "sad" -> Nord9          // Frost Blue
    "angry" -> Nord11       // Aurora Red
    "fearful" -> Nord15     // Aurora Purple
    "disgust" -> Nord14     // Aurora Green
    "surprised" -> Nord12   // Aurora Orange
    else -> Color.Gray
}
