#include "PluginProcessor.h"
#include "../Models/ProjectSerializer.h"
#include "../UI/MainComponent.h"
#include "../Utils/Localization.h"
#include "PluginEditor.h"

HachiTuneAudioProcessor::HachiTuneAudioProcessor()
#ifndef JucePlugin_PreferredChannelConfigurations
    : AudioProcessor(BusesProperties()
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

void HachiTuneAudioProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    hostSampleRate = sampleRate;
    realtimeProcessor.prepareToPlay(sampleRate, samplesPerBlock);

#if JucePlugin_Enable_ARA
    prepareToPlayForARA(sampleRate, samplesPerBlock,
                        getMainBusNumOutputChannels(), getProcessingPrecision());
#endif

    // Pre-allocate capture buffer for non-ARA mode
    int maxSamples = static_cast<int>(sampleRate * MAX_CAPTURE_SECONDS);
    captureBuffer.setSize(getMainBusNumOutputChannels(), maxSamples);
    captureBuffer.clear();
    capturePosition = 0;
    captureState = CaptureState::WaitingForAudio;
}

void HachiTuneAudioProcessor::releaseResources() {
#if JucePlugin_Enable_ARA
    releaseResourcesForARA();
#endif
}

#if !JucePlugin_PreferredChannelConfigurations
bool HachiTuneAudioProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    if (layouts.getMainOutputChannelSet() != layouts.getMainInputChannelSet())
        return false;
    auto out = layouts.getMainOutputChannelSet();
    return out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
}
#endif

bool HachiTuneAudioProcessor::isARAModeActive() const {
#if JucePlugin_Enable_ARA
    if (auto* editor = getActiveEditor()) {
        if (auto* araEditor = dynamic_cast<juce::AudioProcessorEditorARAExtension*>(editor)) {
            if (auto* editorView = araEditor->getARAEditorView()) {
                return editorView->getDocumentController() != nullptr;
            }
        }
    }
#endif
    return false;
}

HostCompatibility::HostInfo HachiTuneAudioProcessor::getHostInfo() const {
    return HostCompatibility::detectHost(const_cast<HachiTuneAudioProcessor*>(this));
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

void HachiTuneAudioProcessor::processBlock(juce::AudioBuffer<float>& buffer,
                                              juce::MidiBuffer& midiMessages) {
    juce::ignoreUnused(midiMessages);
    juce::ScopedNoDenormals noDenormals;

#if JucePlugin_Enable_ARA
    // ARA mode: let ARA renderer handle audio
    if (processBlockForARA(buffer, isRealtime(), getPlayHead()))
        return;
#endif

    // Non-ARA mode
    juce::AudioPlayHead::PositionInfo posInfo;
    if (auto* playHead = getPlayHead()) {
        if (auto info = playHead->getPosition())
            posInfo = *info;
    }

    processNonARAMode(buffer, posInfo);
}

void HachiTuneAudioProcessor::processNonARAMode(juce::AudioBuffer<float>& buffer,
                                                   const juce::AudioPlayHead::PositionInfo& posInfo) {
    const int numSamples = buffer.getNumSamples();
    const int numChannels = buffer.getNumChannels();
    const bool hostIsPlaying = posInfo.getIsPlaying();

    // Update UI cursor position from host playback position (only when we have analyzed audio)
    if (mainComponent) {
        if (hostIsPlaying && captureState == CaptureState::Complete) {
            // Only sync cursor after capture is complete and analyzed
            double timeInSeconds = 0.0;
            if (auto time = posInfo.getTimeInSeconds())
                timeInSeconds = *time;
            else if (auto samples = posInfo.getTimeInSamples())
                timeInSeconds = static_cast<double>(*samples) / hostSampleRate;

            mainComponent->updatePlaybackPosition(timeInSeconds);
        } else if (!hostIsPlaying && captureState == CaptureState::Complete) {
            mainComponent->notifyHostStopped();
        }
    }

    // Check if we have analyzed project ready for real-time processing
    bool hasProject = mainComponent && mainComponent->getProject() &&
                      mainComponent->getProject()->getAudioData().waveform.getNumSamples() > 0 &&
                      !mainComponent->getProject()->getAudioData().f0.empty();

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
    CaptureState state = captureState.load();

    // Start capturing when host plays and we detect audio
    if (state == CaptureState::WaitingForAudio && hostIsPlaying) {
        // Detect audio input
        float maxLevel = 0.0f;
        for (int ch = 0; ch < numChannels; ++ch) {
            auto* data = buffer.getReadPointer(ch);
            for (int i = 0; i < numSamples; ++i)
                maxLevel = std::max(maxLevel, std::abs(data[i]));
        }

        if (maxLevel > AUDIO_THRESHOLD) {
            captureState = CaptureState::Capturing;
            capturePosition = 0;
            state = CaptureState::Capturing;

            // Notify UI that capture started
            if (mainComponent) {
                juce::MessageManager::callAsync([this]() {
                    if (mainComponent)
                        mainComponent->getToolbar().setStatusMessage(TR("progress.recording"));
                });
            }
        }
    }

    if (state == CaptureState::Capturing) {
        if (hostIsPlaying) {
            // Continue capturing while host is playing
            int spaceLeft = captureBuffer.getNumSamples() - capturePosition;
            int toCopy = std::min(numSamples, spaceLeft);

            if (toCopy > 0) {
                for (int ch = 0; ch < std::min(numChannels, captureBuffer.getNumChannels()); ++ch)
                    captureBuffer.copyFrom(ch, capturePosition, buffer, ch, 0, toCopy);
                capturePosition += toCopy;
            }

            // Only stop if buffer is completely full (safety limit)
            if (capturePosition >= captureBuffer.getNumSamples())
                finishCapture();
        } else {
            // Host stopped playing - finish capture and analyze
            finishCapture();
        }
    }

    // Passthrough during capture
}

void HachiTuneAudioProcessor::finishCapture() {
    if (capturePosition < static_cast<int>(hostSampleRate * 0.5))
        return; // Too short

    captureState = CaptureState::Complete;

    // Trim buffer
    juce::AudioBuffer<float> trimmed;
    trimmed.setSize(captureBuffer.getNumChannels(), capturePosition);
    for (int ch = 0; ch < captureBuffer.getNumChannels(); ++ch)
        trimmed.copyFrom(ch, 0, captureBuffer, ch, 0, capturePosition);

    // Send to MainComponent for analysis
    double sr = hostSampleRate;
    juce::MessageManager::callAsync([this, trimmed, sr]() {
        if (mainComponent) {
            mainComponent->getToolbar().setStatusMessage(TR("progress.analyzing"));
            mainComponent->setHostAudio(trimmed, sr);
        }
    });
}

void HachiTuneAudioProcessor::startCapture() {
    captureBuffer.clear();
    capturePosition = 0;
    captureState = CaptureState::Capturing;
}

void HachiTuneAudioProcessor::stopCapture() {
    if (captureState == CaptureState::Capturing)
        finishCapture();
}

void HachiTuneAudioProcessor::setMainComponent(MainComponent* mc) {
    mainComponent = mc;
    if (mc) {
        realtimeProcessor.setProject(mc->getProject());
        realtimeProcessor.setVocoder(mc->getVocoder());
    } else {
        realtimeProcessor.setProject(nullptr);
        realtimeProcessor.setVocoder(nullptr);
    }
}

juce::AudioProcessorEditor* HachiTuneAudioProcessor::createEditor() {
    return new HachiTuneAudioProcessorEditor(*this);
}

void HachiTuneAudioProcessor::getStateInformation(juce::MemoryBlock& destData) {
    if (mainComponent && mainComponent->getProject()) {
        auto json = ProjectSerializer::toJson(*mainComponent->getProject());
        auto jsonString = juce::JSON::toString(json, false);
        destData.append(jsonString.toRawUTF8(), jsonString.getNumBytesAsUTF8());
    }
}

void HachiTuneAudioProcessor::setStateInformation(const void* data, int sizeInBytes) {
    if (mainComponent && mainComponent->getProject()) {
        juce::String jsonString(juce::CharPointer_UTF8(static_cast<const char*>(data)),
                                static_cast<size_t>(sizeInBytes));
        auto json = juce::JSON::parse(jsonString);
        if (json.isObject()) {
            ProjectSerializer::fromJson(*mainComponent->getProject(), json);
        }
    }
}

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() {
    return new HachiTuneAudioProcessor();
}

#if JucePlugin_Enable_ARA
#include "ARADocumentController.h"

const ARA::ARAFactory* JUCE_CALLTYPE createARAFactory() {
    return juce::ARADocumentControllerSpecialisation::createARAFactory<HachiTuneDocumentController>();
}
#endif
