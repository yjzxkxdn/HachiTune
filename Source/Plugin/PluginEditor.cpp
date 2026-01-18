#include "PluginEditor.h"
#include "HostCompatibility.h"
#include "../UI/StyledComponents.h"

#if JucePlugin_Enable_ARA
#include "ARADocumentController.h"
#endif

HachiTuneAudioProcessorEditor::HachiTuneAudioProcessorEditor(HachiTuneAudioProcessor& p)
    : AudioProcessorEditor(&p)
    , audioProcessor(p)
#if JucePlugin_Enable_ARA
    , AudioProcessorEditorARAExtension(&p)
#endif
{
    // Initialize app font
    AppFont::initialize();

    addAndMakeVisible(mainComponent);
    audioProcessor.setMainComponent(&mainComponent);

#if JucePlugin_Enable_ARA
    setupARAMode();
#else
    setupNonARAMode();
#endif

    setupCallbacks();

    setSize(1400, 900);
    setResizable(true, true);
}

HachiTuneAudioProcessorEditor::~HachiTuneAudioProcessorEditor() {
    audioProcessor.setMainComponent(nullptr);
    AppFont::shutdown();  // Release font resources (reference counted)
}

void HachiTuneAudioProcessorEditor::setupARAMode() {
#if JucePlugin_Enable_ARA
    mainComponent.getToolbar().setARAMode(true);

    auto* editorView = getARAEditorView();
    if (!editorView) {
        setupNonARAMode();
        return;
    }

    auto* docController = editorView->getDocumentController();
    if (!docController) {
        setupNonARAMode();
        return;
    }

    auto* pitchDocController = juce::ARADocumentControllerSpecialisation::
        getSpecialisedDocumentController<HachiTuneDocumentController>(docController);

    if (!pitchDocController) {
        setupNonARAMode();
        return;
    }

    // Connect ARA controller to UI
    pitchDocController->setMainComponent(&mainComponent);
    pitchDocController->setRealtimeProcessor(&audioProcessor.getRealtimeProcessor());

    // Setup re-analyze callback
    mainComponent.onReanalyzeRequested = [pitchDocController]() {
        pitchDocController->reanalyze();
    };

    // Check for existing audio sources
    auto* juceDocument = docController->getDocument();
    if (juceDocument) {
        auto& audioSources = juceDocument->getAudioSources<juce::ARAAudioSource>();

        if (!audioSources.empty()) {
            // Process first audio source
            auto* source = audioSources.front();
            if (source && source->getSampleCount() > 0) {
                juce::ARAAudioSourceReader reader(source);
                int numSamples = static_cast<int>(source->getSampleCount());
                int numChannels = source->getChannelCount();
                double sampleRate = source->getSampleRate();

                juce::AudioBuffer<float> buffer(numChannels, numSamples);
                if (reader.read(&buffer, 0, numSamples, 0, true, true)) {
                    mainComponent.setHostAudio(buffer, sampleRate);
                    return;
                }
            }
        }
    }
#endif
}

void HachiTuneAudioProcessorEditor::setupNonARAMode() {
    mainComponent.getToolbar().setARAMode(false);
}

void HachiTuneAudioProcessorEditor::setupCallbacks() {
    // When project data changes (analysis complete or synthesis complete)
    mainComponent.onProjectDataChanged = [this]() {
        // IMPORTANT: Set project FIRST, then vocoder
        // setProject() calls invalidate() which needs a valid project
        if (mainComponent.getProject())
            audioProcessor.getRealtimeProcessor().setProject(mainComponent.getProject());

        // setVocoder() does NOT call invalidate() anymore (safe to call anytime)
        if (mainComponent.getVocoder())
            audioProcessor.getRealtimeProcessor().setVocoder(mainComponent.getVocoder());

        // No need for extra invalidate() - setProject() already calls it
    };

    // onPitchEditFinished is handled by onProjectDataChanged (called after async synthesis completes)
    // No need for separate callback here
}

void HachiTuneAudioProcessorEditor::paint(juce::Graphics&) {
    // MainComponent handles all painting
}

void HachiTuneAudioProcessorEditor::resized() {
    mainComponent.setBounds(getLocalBounds());
}
