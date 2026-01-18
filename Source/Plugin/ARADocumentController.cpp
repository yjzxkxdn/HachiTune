#include "ARADocumentController.h"

#if JucePlugin_Enable_ARA

#include "../Models/ProjectSerializer.h"
#include "../UI/MainComponent.h"

//==============================================================================
// HachiTunePlaybackRenderer
//==============================================================================

HachiTuneDocumentController* HachiTunePlaybackRenderer::getDocController() const {
    auto* docController = getDocumentController();
    return juce::ARADocumentControllerSpecialisation::
        getSpecialisedDocumentController<HachiTuneDocumentController>(docController);
}

void HachiTunePlaybackRenderer::prepareToPlay(double sampleRateIn, int maxBlockSize,
                                                 int numChannelsIn,
                                                 juce::AudioProcessor::ProcessingPrecision,
                                                 AlwaysNonRealtime alwaysNonRealtime) {
    sampleRate = sampleRateIn;
    numChannels = numChannelsIn;
    tempBuffer = std::make_unique<juce::AudioBuffer<float>>(numChannels, maxBlockSize);

    bool useBuffered = (alwaysNonRealtime == AlwaysNonRealtime::no);
    juce::ignoreUnused(useBuffered);

    // Create readers for all playback regions
    for (auto* region : getPlaybackRegions()) {
        auto* source = region->getAudioModification()->getAudioSource();
        if (readers.find(source) == readers.end())
            readers.emplace(source, std::make_unique<juce::ARAAudioSourceReader>(source));
    }
}

void HachiTunePlaybackRenderer::releaseResources() {
    readers.clear();
    tempBuffer.reset();
}

bool HachiTunePlaybackRenderer::readFromARARegions(juce::AudioBuffer<float>& buffer,
                                                      juce::int64 timeInSamples,
                                                      int numSamples) {
    bool didRender = false;
    auto blockRange = juce::Range<juce::int64>::withStartAndLength(timeInSamples, numSamples);

    for (auto* region : getPlaybackRegions()) {
        auto playbackRange = region->getSampleRange(sampleRate,
            juce::ARAPlaybackRegion::IncludeHeadAndTail::no);
        auto renderRange = blockRange.getIntersectionWith(playbackRange);

        if (renderRange.isEmpty())
            continue;

        // Get modification range
        juce::Range<juce::int64> modRange{
            region->getStartInAudioModificationSamples(),
            region->getEndInAudioModificationSamples()
        };
        auto modOffset = modRange.getStart() - playbackRange.getStart();
        renderRange = renderRange.getIntersectionWith(modRange.movedToStartAt(playbackRange.getStart()));

        if (renderRange.isEmpty())
            continue;

        // Get reader
        auto* source = region->getAudioModification()->getAudioSource();
        auto it = readers.find(const_cast<juce::ARAAudioSource*>(source));
        if (it == readers.end())
            continue;

        int samplesToRead = static_cast<int>(renderRange.getLength());
        int bufferOffset = static_cast<int>(renderRange.getStart() - blockRange.getStart());
        auto sourceStart = renderRange.getStart() + modOffset;

        auto& readBuffer = didRender ? *tempBuffer : buffer;
        if (!it->second->read(&readBuffer, bufferOffset, samplesToRead, sourceStart, true, true))
            continue;

        if (didRender) {
            // Mix with existing
            for (int ch = 0; ch < numChannels; ++ch)
                buffer.addFrom(ch, bufferOffset, *tempBuffer, ch, bufferOffset, samplesToRead);
        } else {
            // Clear areas outside render range
            if (bufferOffset > 0)
                buffer.clear(0, bufferOffset);
            int endOffset = bufferOffset + samplesToRead;
            if (endOffset < numSamples)
                buffer.clear(endOffset, numSamples - endOffset);
            didRender = true;
        }
    }

    return didRender;
}

bool HachiTunePlaybackRenderer::processBlock(juce::AudioBuffer<float>& buffer,
                                                juce::AudioProcessor::Realtime,
                                                const juce::AudioPlayHead::PositionInfo& posInfo) noexcept {
    auto timeInSamples = posInfo.getTimeInSamples().orFallback(0);
    bool isPlaying = posInfo.getIsPlaying();
    int numSamples = buffer.getNumSamples();

    // Get document controller for accessing MainComponent
    auto* docCtrl = getDocController();

    // Update UI cursor position from host playback position
    if (docCtrl && docCtrl->getMainComponent()) {
        if (isPlaying) {
            double timeInSeconds = static_cast<double>(timeInSamples) / sampleRate;
            docCtrl->getMainComponent()->updatePlaybackPosition(timeInSeconds);
        } else {
            docCtrl->getMainComponent()->notifyHostStopped();
        }
    }

    if (!isPlaying) {
        buffer.clear();
        return true;
    }

    // Read from ARA regions
    juce::AudioBuffer<float> inputBuffer(buffer.getNumChannels(), numSamples);
    bool didRender = readFromARARegions(inputBuffer, timeInSamples, numSamples);

    if (!didRender) {
        buffer.clear();
        return true;
    }

    // Get processor from document controller (dynamic lookup)
    auto* realtimeProcessor = docCtrl ? docCtrl->getRealtimeProcessor() : nullptr;

    // Apply pitch correction if processor available and ready
    if (realtimeProcessor && realtimeProcessor->isReady()) {
        if (realtimeProcessor->processBlock(inputBuffer, buffer, &posInfo)) {
            return true;
        } else {
            DBG("ARA processBlock: realtimeProcessor->processBlock returned false (passthrough)");
        }
    } else {
        // Log why we're not using the processor
        if (!realtimeProcessor) {
            DBG("ARA processBlock: realtimeProcessor is null");
        } else if (!realtimeProcessor->isReady()) {
            DBG("ARA processBlock: realtimeProcessor not ready");
        }
    }

    // Fallback: copy input to output
    buffer.makeCopyOf(inputBuffer);
    return true;
}

//==============================================================================
// HachiTuneDocumentController
//==============================================================================

void HachiTuneDocumentController::processAudioSource(juce::ARAAudioSource* source) {
    if (!mainComponent || !source)
        return;

    auto numSamples = static_cast<int>(source->getSampleCount());
    auto numChannels = source->getChannelCount();
    auto sourceSampleRate = source->getSampleRate();

    if (numSamples <= 0 || numChannels <= 0 || sourceSampleRate <= 0)
        return;

    juce::ARAAudioSourceReader reader(source);
    juce::AudioBuffer<float> buffer(numChannels, numSamples);

    if (!reader.read(&buffer, 0, numSamples, 0, true, true))
        return;

    mainComponent->getToolbar().setStatusMessage("ARA Mode - Analyzing...");
    mainComponent->setHostAudio(buffer, sourceSampleRate);
}

void HachiTuneDocumentController::didAddAudioSourceToDocument(juce::ARADocument*,
                                                                 juce::ARAAudioSource* audioSource) {
    currentAudioSource = audioSource;
    processAudioSource(audioSource);
}

void HachiTuneDocumentController::reanalyze() {
    if (currentAudioSource)
        processAudioSource(currentAudioSource);
}

juce::ARAPlaybackRenderer* HachiTuneDocumentController::doCreatePlaybackRenderer() noexcept {
    return new HachiTunePlaybackRenderer(
        ARADocumentControllerSpecialisation::getDocumentController());
}

bool HachiTuneDocumentController::doRestoreObjectsFromStream(juce::ARAInputStream& input,
                                                                const juce::ARARestoreObjectsFilter*) noexcept {
    auto dataSize = input.readInt64();
    if (dataSize <= 0)
        return true;

    juce::MemoryBlock data;
    data.setSize(static_cast<size_t>(dataSize));
    input.read(data.getData(), static_cast<int>(dataSize));

    if (mainComponent && mainComponent->getProject()) {
        juce::String jsonString(juce::CharPointer_UTF8(static_cast<const char*>(data.getData())),
                                data.getSize());
        auto json = juce::JSON::parse(jsonString);
        if (json.isObject()) {
            ProjectSerializer::fromJson(*mainComponent->getProject(), json);
        }
    }

    return !input.failed();
}

bool HachiTuneDocumentController::doStoreObjectsToStream(juce::ARAOutputStream& output,
                                                            const juce::ARAStoreObjectsFilter*) noexcept {
    if (!mainComponent || !mainComponent->getProject()) {
        output.writeInt64(0);
        return true;
    }

    auto json = ProjectSerializer::toJson(*mainComponent->getProject());
    auto jsonString = juce::JSON::toString(json, false);

    output.writeInt64(static_cast<juce::int64>(jsonString.getNumBytesAsUTF8()));
    return output.write(jsonString.toRawUTF8(), static_cast<int>(jsonString.getNumBytesAsUTF8()));
}

#endif // JucePlugin_Enable_ARA
