package com.example.voxsense.ui.main

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.voxsense.data.DataRepository
import com.example.voxsense.data.EmotionClassifier
import com.example.voxsense.data.EmotionPrediction
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

import com.example.voxsense.data.AppModel
import com.example.voxsense.data.ModelManager
import com.example.voxsense.data.DownloadState

data class ModelUIState(
    val model: AppModel,
    val isDownloaded: Boolean,
    val isActive: Boolean,
    val isDownloading: Boolean,
    val downloadProgress: Int?
)

class MainScreenViewModel(private val dataRepository: DataRepository) : ViewModel() {
    
    private val modelManager = dataRepository.modelManager

    private val _downloadingModelId = MutableStateFlow<String?>(null)
    val downloadingModelId: StateFlow<String?> = _downloadingModelId.asStateFlow()

    private val _downloadProgress = MutableStateFlow<Int?>(null)
    val downloadProgress: StateFlow<Int?> = _downloadProgress.asStateFlow()

    private val _downloadError = MutableStateFlow<String?>(null)
    val downloadError: StateFlow<String?> = _downloadError.asStateFlow()

    private val _modelStatesVersion = MutableStateFlow(0)

    val modelStates: StateFlow<List<ModelUIState>> = combine(
        _downloadingModelId,
        _downloadProgress,
        _modelStatesVersion
    ) { downloadingId, progress, _ ->
        ModelManager.AVAILABLE_MODELS.map { model ->
            ModelUIState(
                model = model,
                isDownloaded = modelManager.isModelDownloaded(model.id),
                isActive = modelManager.getActiveModelId() == model.id,
                isDownloading = downloadingId == model.id,
                downloadProgress = if (downloadingId == model.id) progress else null
            )
        }
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = ModelManager.AVAILABLE_MODELS.map { model ->
            ModelUIState(
                model = model,
                isDownloaded = modelManager.isModelDownloaded(model.id),
                isActive = modelManager.getActiveModelId() == model.id,
                isDownloading = false,
                downloadProgress = null
            )
        }
    )
    
    private val _isRecording = MutableStateFlow(false)
    val isRecording: StateFlow<Boolean> = _isRecording.asStateFlow()

    private val _isProcessing = MutableStateFlow(false)
    val isProcessing: StateFlow<Boolean> = _isProcessing.asStateFlow()

    private val _currentAmplitude = MutableStateFlow(0f)
    val currentAmplitude: StateFlow<Float> = _currentAmplitude.asStateFlow()

    private val _recordingDuration = MutableStateFlow(0L) // in milliseconds
    val recordingDuration: StateFlow<Long> = _recordingDuration.asStateFlow()

    private val _enabledEmotions = MutableStateFlow<Set<String>>(EmotionClassifier.EMOTIONS.toSet())
    val enabledEmotions: StateFlow<Set<String>> = _enabledEmotions.asStateFlow()

    private val _usePreEmphasis = MutableStateFlow(true)
    val usePreEmphasis: StateFlow<Boolean> = _usePreEmphasis.asStateFlow()

    private val _useNoiseGate = MutableStateFlow(false)
    val useNoiseGate: StateFlow<Boolean> = _useNoiseGate.asStateFlow()

    private val _recordingLimitSeconds = MutableStateFlow<Int?>(null)
    val recordingLimitSeconds: StateFlow<Int?> = _recordingLimitSeconds.asStateFlow()

    private var recordingStartTime = 0L
    private var timerJob: Job? = null

    val uiState: StateFlow<MainScreenUiState> = combine(
        dataRepository.latestPrediction,
        dataRepository.history,
        _isRecording,
        _isProcessing,
        _enabledEmotions,
        _usePreEmphasis,
        _useNoiseGate,
        _recordingLimitSeconds
    ) { array ->
        @Suppress("UNCHECKED_CAST")
        val latest = array[0] as EmotionPrediction?
        @Suppress("UNCHECKED_CAST")
        val history = array[1] as List<EmotionPrediction>
        val recording = array[2] as Boolean
        val processing = array[3] as Boolean
        @Suppress("UNCHECKED_CAST")
        val emotions = array[4] as Set<String>
        val preEmp = array[5] as Boolean
        val noiseGate = array[6] as Boolean
        val limit = array[7] as Int?

        MainScreenUiState.Success(
            latestPrediction = latest,
            history = history,
            isRecording = recording,
            isProcessing = processing,
            enabledEmotions = emotions,
            usePreEmphasis = preEmp,
            useNoiseGate = noiseGate,
            recordingLimitSeconds = limit
        )
    }.stateIn(
        scope = viewModelScope,
        started = SharingStarted.WhileSubscribed(5000),
        initialValue = MainScreenUiState.Success(
            latestPrediction = null,
            history = emptyList(),
            isRecording = false,
            isProcessing = false,
            enabledEmotions = EmotionClassifier.EMOTIONS.toSet(),
            usePreEmphasis = true,
            useNoiseGate = false,
            recordingLimitSeconds = null
        )
    )

    fun startRecording() {
        if (_isRecording.value || _isProcessing.value) return
        
        _currentAmplitude.value = 0f
        _recordingDuration.value = 0L
        
        val success = dataRepository.startRecording { amplitude ->
            _currentAmplitude.value = amplitude
        }
        
        if (success) {
            _isRecording.value = true
            recordingStartTime = System.currentTimeMillis()
            startTimer()
        }
    }

    fun stopRecording() {
        if (!_isRecording.value) return
        
        stopTimer()
        _isRecording.value = false
        _isProcessing.value = true
        
        viewModelScope.launch {
            try {
                dataRepository.stopRecording(
                    usePreEmphasis = _usePreEmphasis.value,
                    useNoiseGate = _useNoiseGate.value
                )
            } catch (e: Exception) {
                e.printStackTrace()
            } finally {
                _isProcessing.value = false
                _currentAmplitude.value = 0f
            }
        }
    }

    fun clearHistory() {
        dataRepository.clearHistory()
    }

    fun togglePreEmphasis() {
        _usePreEmphasis.value = !_usePreEmphasis.value
    }

    fun toggleNoiseGate() {
        _useNoiseGate.value = !_useNoiseGate.value
    }

    fun setRecordingLimit(seconds: Int?) {
        _recordingLimitSeconds.value = seconds
    }

    fun toggleEmotion(emotion: String) {
        val current = _enabledEmotions.value.toMutableSet()
        if (current.contains(emotion)) {
            // Keep at least one emotion enabled to avoid empty breakdowns
            if (current.size > 1) {
                current.remove(emotion)
            }
        } else {
            current.add(emotion)
        }
        _enabledEmotions.value = current
    }

    private fun startTimer() {
        timerJob?.cancel()
        timerJob = viewModelScope.launch {
            while (_isRecording.value) {
                val elapsed = System.currentTimeMillis() - recordingStartTime
                _recordingDuration.value = elapsed
                
                val limit = _recordingLimitSeconds.value
                if (limit != null && elapsed >= limit * 1000L) {
                    stopRecording()
                }
                delay(100)
            }
        }
    }

    fun selectModel(modelId: String) {
        if (modelManager.isModelDownloaded(modelId)) {
            modelManager.setActiveModelId(modelId)
            _modelStatesVersion.value += 1
        }
    }

    fun deleteModel(modelId: String) {
        if (modelManager.deleteModel(modelId)) {
            _modelStatesVersion.value += 1
        }
    }

    fun downloadModel(modelId: String) {
        if (_downloadingModelId.value != null) return // Only allow one download at a time
        
        _downloadingModelId.value = modelId
        _downloadProgress.value = 0
        _downloadError.value = null
        
        viewModelScope.launch {
            modelManager.downloadModel(modelId).collect { state ->
                when (state) {
                    is DownloadState.Idle -> {}
                    is DownloadState.Loading -> {
                        _downloadProgress.value = 0
                    }
                    is DownloadState.Progress -> {
                        _downloadProgress.value = state.percentage
                    }
                    is DownloadState.Success -> {
                        _downloadingModelId.value = null
                        _downloadProgress.value = null
                        _modelStatesVersion.value += 1
                    }
                    is DownloadState.Error -> {
                        _downloadingModelId.value = null
                        _downloadProgress.value = null
                        _downloadError.value = state.message
                    }
                }
            }
        }
    }
    
    fun clearDownloadError() {
        _downloadError.value = null
    }

    private fun stopTimer() {
        timerJob?.cancel()
        timerJob = null
    }

    override fun onCleared() {
        super.onCleared()
        stopTimer()
    }
}

sealed interface MainScreenUiState {
    data class Success(
        val latestPrediction: EmotionPrediction?,
        val history: List<EmotionPrediction>,
        val isRecording: Boolean,
        val isProcessing: Boolean,
        val enabledEmotions: Set<String>,
        val usePreEmphasis: Boolean,
        val useNoiseGate: Boolean,
        val recordingLimitSeconds: Int?
    ) : MainScreenUiState
    
    data class Error(val throwable: Throwable) : MainScreenUiState
}
