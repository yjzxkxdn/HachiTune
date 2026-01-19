#include "PluginProcessor.h"
#include "../Models/ProjectSerializer.h"
#include "../UI/MainComponent.h"
#include "../Utils/Localization.h"
#include "PluginEditor.h"

HachiTuneAudioProcessor::HachiTuneAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(
          BusesProperties()
              .withInput("Input", juce::AudioChannelSet::stereo(), true)
              .withOutput("Output", juce::AudioChannelSet::stereo(), true))
#endif
{
}

HachiTuneAudioProcessor::~HachiTuneAudioProcessor() = default;

const juce::String HachiTuneAudioProcessor::getName() const {
  return JucePlugin_Name;
}

bool HachiTuneAudioProcessor::acceptsMidi() const {
#if JucePlugin_WantsMidiInput
  return true;
#else
  return false;
#endif
}

bool HachiTuneAudioProcessor::producesMidi() const {
#if JucePlugin_ProducesMidiOutput
  return true;
#else
  return false;
#endif
}

bool HachiTuneAudioProcessor::isMidiEffect() const {
#if JucePlugin_IsMidiEffect
  return true;
#else
  return false;
#endif
}

void HachiTuneAudioProcessor::prepareToPlay(double sampleRate,
                                            int samplesPerBlock) {
  hostSampleRate = sampleRate;
  realtimeProcessor.prepareToPlay(sampleRate, samplesPerBlock);

#if JucePlugin_Enable_ARA
  prepareToPlayForARA(sampleRate, samplesPerBlock,
                      getMainBusNumOutputChannels(), getProcessingPrecision());
#endif

  // Non-ARA capture controller
  captureController->prepare(sampleRate, getMainBusNumOutputChannels(),
                             MAX_CAPTURE_SECONDS);
  lastCaptureUiState = captureController->getState();
}

void HachiTuneAudioProcessor::releaseResources() {
#if JucePlugin_Enable_ARA
  releaseResourcesForARA();
#endif
}

#if !JucePlugin_PreferredChannelConfigurations
bool HachiTuneAudioProcessor::isBusesLayoutSupported(
    const BusesLayout &layouts) const {
  if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
    return false;
  auto out = layouts.getMainOutputChannelSet();
  return out == juce::AudioChannelSet::mono() ||
         out == juce::AudioChannelSet::stereo();
}
#endif

bool HachiTuneAudioProcessor::isARAModeActive() const {
#if JucePlugin_Enable_ARA
  if (auto *editor = getActiveEditor()) {
    if (auto *araEditor =
            dynamic_cast<juce::AudioProcessorEditorARAExtension *>(editor)) {
      if (auto *editorView = araEditor->getARAEditorView()) {
        return editorView->getDocumentController() != nullptr;
      }
    }
  }
#endif
  return false;
}

HostCompatibility::HostInfo HachiTuneAudioProcessor::getHostInfo() const {
  return HostCompatibility::detectHost(
      const_cast<HachiTuneAudioProcessor *>(this));
}

juce::String HachiTuneAudioProcessor::getHostStatusMessage() const {
  auto hostInfo = getHostInfo();
  bool araActive = isARAModeActive();

  if (hostInfo.type != HostCompatibility::HostType::Unknown) {
    if (araActive)
      return hostInfo.name + " - ARA Mode";
    if (hostInfo.supportsARA)
      return hostInfo.name + " - Non-ARA (ARA Available)";
    return hostInfo.name + " - Non-ARA Mode";
  }
  return araActive ? "ARA Mode" : "Non-ARA Mode";
}

void HachiTuneAudioProcessor::processBlock(juce::AudioBuffer<float> &buffer,
                                           juce::MidiBuffer &midiMessages) {
  juce::ignoreUnused(midiMessages);
  juce::ScopedNoDenormals noDenormals;

  // Best-effort host transport control (must be called from processBlock)
  if (auto *playHead = getPlayHead()) {
    if (playHead->canControlTransport()) {
      if (stopRequested.exchange(false)) {
        playHead->transportPlay(false);
        playHead->transportRewind();
      }

      if (hasPendingPlayRequest.exchange(false)) {
        playHead->transportPlay(requestedPlayState.load());
      }
    }
  }

#if JucePlugin_Enable_ARA
  // ARA mode: let ARA renderer handle audio
  if (processBlockForARA(buffer, isRealtime(), getPlayHead()))
    return;
#endif

  // Non-ARA mode
  juce::AudioPlayHead::PositionInfo posInfo;
  if (auto *playHead = getPlayHead()) {
    if (auto info = playHead->getPosition())
      posInfo = *info;
  }

  processNonARAMode(buffer, posInfo,
                    isRealtime() == juce::AudioProcessor::Realtime::yes);
}

void HachiTuneAudioProcessor::processNonARAMode(
    juce::AudioBuffer<float> &buffer,
    const juce::AudioPlayHead::PositionInfo &posInfo, bool isRealtime) {
  const int numSamples = buffer.getNumSamples();
  const int numChannels = buffer.getNumChannels();
  const bool hostIsPlaying = posInfo.getIsPlaying();

  // Check if we have analyzed project ready for real-time processing
  bool hasProject =
      mainComponent && mainComponent->getProject() &&
      mainComponent->getProject()->getAudioData().waveform.getNumSamples() >
          0 &&
      !mainComponent->getProject()->getAudioData().f0.empty();

  // Update UI cursor position from host playback position (only when we have
  // analyzed audio)
  if (isRealtime && mainComponent) {
    if (hostIsPlaying && hasProject) {
      // Only sync cursor after capture is complete and analyzed
      double timeInSeconds = 0.0;
      if (auto samples = posInfo.getTimeInSamples())
        timeInSeconds = static_cast<double>(*samples) / hostSampleRate;
      else if (auto time = posInfo.getTimeInSeconds())
        timeInSeconds = *time;

      auto state = hostUiSyncState;
      state->latestSeconds.store(timeInSeconds);

      // Never touch UI on the audio thread: coalesce to a single async update
      if (!state->posPending.exchange(true)) {
        juce::Component::SafePointer<MainComponent> safeMain(mainComponent);
        juce::MessageManager::callAsync([safeMain, state]() {
          state->posPending.store(false);
          if (safeMain != nullptr)
            safeMain->updatePlaybackPosition(state->latestSeconds.load());
        });
      }
    } else if (!hostIsPlaying && hasProject) {
      auto state = hostUiSyncState;
      if (!state->stoppedPending.exchange(true)) {
        juce::Component::SafePointer<MainComponent> safeMain(mainComponent);
        juce::MessageManager::callAsync([safeMain, state]() {
          state->stoppedPending.store(false);
          if (safeMain != nullptr)
            safeMain->notifyHostStopped();
        });
      }
    }
  }

  if (!hostIsPlaying) {
    // Still let the capture state machine observe transport stop so it can
    // finalize and dispatch analysis, but never output audio when stopped.
    captureController->processBlock(buffer, false);

    if (captureController->shouldFinalize()) {
      NonAraCaptureController::FinalizeResult result;
      if (captureController->finalizeCapture(hostSampleRate, result) &&
          mainComponent) {
        juce::Component::SafePointer<MainComponent> safeMain(mainComponent);
        auto controller = captureController;
        juce::MessageManager::callAsync([safeMain, controller,
                                         samples = result.numSamples,
                                         sr = result.sampleRate]() mutable {
          if (safeMain == nullptr)
            return;
          if (!controller)
            return;
          auto trimmed = controller->copyCapturedAudio(samples);
          controller->onAnalysisDispatched();
          safeMain->getToolbar().setStatusMessage(TR("progress.analyzing"));
          safeMain->setHostAudio(trimmed, sr);
        });
      }
    }

    buffer.clear();
    return;
  }

  if (hasProject && realtimeProcessor.isReady()) {
    // Real-time pitch correction mode
    juce::AudioBuffer<float> outputBuffer(numChannels, numSamples);
    if (realtimeProcessor.processBlock(buffer, outputBuffer, &posInfo)) {
      for (int ch = 0; ch < numChannels; ++ch)
        buffer.copyFrom(ch, 0, outputBuffer, ch, 0, numSamples);
    }
    return;
  }

  // Capture mode
  captureController->processBlock(buffer, hostIsPlaying);

  // UI: transition into recording
  auto currentState = captureController->getState();
  if (currentState != lastCaptureUiState) {
    if (currentState == NonAraCaptureController::State::Capturing &&
        mainComponent) {
      juce::Component::SafePointer<MainComponent> safeMain(mainComponent);
      juce::MessageManager::callAsync([safeMain]() {
        if (safeMain)
          safeMain->getToolbar().setStatusMessage(TR("progress.recording"));
      });
    }
    lastCaptureUiState = currentState;
  }

  if (captureController->shouldFinalize()) {
    NonAraCaptureController::FinalizeResult result;
    if (captureController->finalizeCapture(hostSampleRate, result) &&
        mainComponent) {
      juce::Component::SafePointer<MainComponent> safeMain(mainComponent);
      auto controller = captureController;
      juce::MessageManager::callAsync([safeMain, controller,
                                       samples = result.numSamples,
                                       sr = result.sampleRate]() mutable {
        if (safeMain == nullptr)
          return;
        if (!controller)
          return;
        auto trimmed = controller->copyCapturedAudio(samples);
        controller->onAnalysisDispatched();
        safeMain->getToolbar().setStatusMessage(TR("progress.analyzing"));
        safeMain->setHostAudio(trimmed, sr);
      });
    }
  }

  // Passthrough during capture
}

void HachiTuneAudioProcessor::startCapture() {
  captureController->resetToWaiting();
}

void HachiTuneAudioProcessor::stopCapture() { captureController->stop(); }

void HachiTuneAudioProcessor::setMainComponent(MainComponent *mc) {
  mainComponent = mc;
  if (mc) {
    realtimeProcessor.setProject(mc->getProject());
    realtimeProcessor.setVocoder(mc->getVocoder());

    if (mc->getProject() && pendingStateJson.isNotEmpty()) {
      auto json = juce::JSON::parse(pendingStateJson);
      if (json.isObject()) {
        ProjectSerializer::fromJson(*mc->getProject(), json);
      }
      pendingStateJson.clear();
    }
  } else {
    realtimeProcessor.setProject(nullptr);
    realtimeProcessor.setVocoder(nullptr);
  }
}

juce::AudioProcessorEditor *HachiTuneAudioProcessor::createEditor() {
  return new HachiTuneAudioProcessorEditor(*this);
}

void HachiTuneAudioProcessor::getStateInformation(juce::MemoryBlock &destData) {
  juce::String jsonString;
  if (mainComponent && mainComponent->getProject()) {
    auto json = ProjectSerializer::toJson(*mainComponent->getProject());
    jsonString = juce::JSON::toString(json, false);
  } else if (pendingStateJson.isNotEmpty()) {
    jsonString = pendingStateJson;
  }
  if (jsonString.isNotEmpty())
    destData.append(jsonString.toRawUTF8(), jsonString.getNumBytesAsUTF8());
}

void HachiTuneAudioProcessor::setStateInformation(const void *data,
                                                  int sizeInBytes) {
  juce::String jsonString(
      juce::CharPointer_UTF8(static_cast<const char *>(data)),
      static_cast<size_t>(sizeInBytes));

  if (mainComponent && mainComponent->getProject()) {
    auto json = juce::JSON::parse(jsonString);
    if (json.isObject()) {
      ProjectSerializer::fromJson(*mainComponent->getProject(), json);
      return;
    }
  }

  pendingStateJson = jsonString;
}

juce::AudioProcessor *JUCE_CALLTYPE createPluginFilter() {
  return new HachiTuneAudioProcessor();
}

#if JucePlugin_Enable_ARA
#include "ARADocumentController.h"

const ARA::ARAFactory *JUCE_CALLTYPE createARAFactory() {
  return juce::ARADocumentControllerSpecialisation::createARAFactory<
      HachiTuneDocumentController>();
}
#endif
