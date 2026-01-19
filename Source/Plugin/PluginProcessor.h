#pragma once

#include "../Audio/RealtimePitchProcessor.h"
#include "../JuceHeader.h"
#include "HostCompatibility.h"
#include "NonAraCaptureController.h"
#include <atomic>
#include <memory>

class MainComponent;

/**
 * HachiTune Audio Processor
 *
 * Supports two modes like Melodyne:
 * 1. ARA Mode: Direct audio access via ARA protocol (Studio One, Cubase, Logic,
 * etc.)
 * 2. Non-ARA Mode: Auto-capture and process (FL Studio, Ableton, etc.)
 */
class HachiTuneAudioProcessor : public juce::AudioProcessor
#if JucePlugin_Enable_ARA
    ,
                                public juce::AudioProcessorARAExtension
#endif
{
public:
  HachiTuneAudioProcessor();
  ~HachiTuneAudioProcessor() override;

  // AudioProcessor interface
  void prepareToPlay(double sampleRate, int samplesPerBlock) override;
  void releaseResources() override;
  void processBlock(juce::AudioBuffer<float> &, juce::MidiBuffer &) override;

#if !JucePlugin_PreferredChannelConfigurations
  bool isBusesLayoutSupported(const BusesLayout &layouts) const override;
#endif

  juce::AudioProcessorEditor *createEditor() override;
  bool hasEditor() const override { return true; }

  const juce::String getName() const override;
  bool acceptsMidi() const override;
  bool producesMidi() const override;
  bool isMidiEffect() const override;
  double getTailLengthSeconds() const override { return 0.0; }

  int getNumPrograms() override { return 1; }
  int getCurrentProgram() override { return 0; }
  void setCurrentProgram(int) override {}
  const juce::String getProgramName(int) override { return {}; }
  void changeProgramName(int, const juce::String &) override {}

  void getStateInformation(juce::MemoryBlock &destData) override;
  void setStateInformation(const void *data, int sizeInBytes) override;

  // Mode detection
  bool isARAModeActive() const;
  HostCompatibility::HostInfo getHostInfo() const;
  juce::String getHostStatusMessage() const;

  // Editor connection
  void setMainComponent(MainComponent *mc);
  MainComponent *getMainComponent() const { return mainComponent; }

  // Host transport control (best-effort; host may ignore)
  void requestHostPlayState(bool shouldPlay) {
    requestedPlayState.store(shouldPlay);
    hasPendingPlayRequest.store(true);
  }
  void requestHostStop() { stopRequested.store(true); }

  // Real-time processor access
  RealtimePitchProcessor &getRealtimeProcessor() { return realtimeProcessor; }
  double getHostSampleRate() const { return hostSampleRate; }

  // Non-ARA mode: capture control
  void startCapture();
  void stopCapture();
  bool isCapturing() const {
    return captureController && captureController->getState() ==
                                    NonAraCaptureController::State::Capturing;
  }

private:
  struct HostUiSyncState {
    std::atomic<double> latestSeconds{0.0};
    std::atomic<bool> posPending{false};
    std::atomic<bool> stoppedPending{false};
  };

  void processNonARAMode(juce::AudioBuffer<float> &buffer,
                         const juce::AudioPlayHead::PositionInfo &posInfo,
                         bool isRealtime);

  RealtimePitchProcessor realtimeProcessor;
  MainComponent *mainComponent = nullptr;
  std::shared_ptr<HostUiSyncState> hostUiSyncState =
      std::make_shared<HostUiSyncState>();
  double hostSampleRate = 44100.0;

  juce::String pendingStateJson;

  // Transport control requests from UI (executed on audio thread)
  std::atomic<bool> requestedPlayState{false};
  std::atomic<bool> hasPendingPlayRequest{false};
  std::atomic<bool> stopRequested{false};

  // Non-ARA capture (Stage 2A): decoupled controller
  std::shared_ptr<NonAraCaptureController> captureController =
      std::make_shared<NonAraCaptureController>();
  NonAraCaptureController::State lastCaptureUiState =
      NonAraCaptureController::State::Idle;
  static constexpr int MAX_CAPTURE_SECONDS = 300; // 5 minutes max

  JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(HachiTuneAudioProcessor)
};
