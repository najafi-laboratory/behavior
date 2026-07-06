# Block Single Interval Discrimination 202606

This protocol runs a short/long interval discrimination task in Bpod with synchronized visual and audio stimulus presentation, moving spouts, reward delivery, probe trials, opto/chemo tagging, and live session plots.

## Update note

### 2026.07.06
- Changed pre reward delay to pre outcome delay.
- Now pre outcome delay is applied before punish.
- Added opto during SpoutInDelay.
- Added more doric params into gui and ask for check.
- Added pre stim delay.
- Now opto stim also cover pre stim delay.
- Adjust gui layouts.
- Isolated water delivery into reward state.
- Separated opto reward and opto post reward.

### 2026.06.25
- Added opto options to spout-in delay and pre-outcome delay.
- Added opto option to punish ITI.
- Adjusted canvas layouts.
- Adjusted printing info after session ends.

### 2026.06.24
- First release.

## Basic Workflow

1. Launch `block_single_interval_discrimination_202606`.
2. Edit the Bpod GUI parameters.
3. Press Enter in MATLAB to configure hardware and generate the session plan.
4. Wait for the stimulus screen to show the grey ready screen.
5. Press Enter again to start trials.
6. End the session from Bpod when finished. The protocol returns the display to grey, closes session windows, stops hardware objects, and saves Bpod data.

## Main Features

- Two trial types: short and long.
- Configurable contingency: short-left/long-right or short-right/long-left.
- Session-level block control with 50/50, short-majority, and long-majority blocks.
- Per-trial ISI sampling from fixed or uniform ranges.
- ITI and punish ITI sampling from manual or bounded exponential distributions.
- Visual-only, audio-only, or audio+visual stimulus modes.
- Optional `image.png` visual stimulus instead of the generated grating.
- Audio and visual stimulus modes use the same total duration and sync-patch timing.
- Moving-spout trained and naive task modes.
- Probe trials:
  - Stimulus-only probe.
  - Servo-only probe.
- Opto trial tags, arbitrary stimulus/spout-in-delay/choice/pre-outcome/post-reward/punish-ITI period combinations, PWM hardware control, trigger settings, and chemo session tagging in the Manipulation GUI section. Opto settings can be changed during the session; the next trial uses the current GUI settings.
- Live plots for trial types, blocks, probe/opto, ISI, outcomes, lick rates, reaction time, state timing, and events.

## GUI Parameter Quick Reference

### Session

- `MaxTrials`: maximum session length.
- `TrainingMode`: naive auto-reward workflow or trained choice workflow.
- `Contingency`: maps short/long trials to left/right target sides.
- `BlockNum`: block structure: 50/50 only, alternating short/long majority, or 50/50/short/long without repeats.
- `WarmupBlockNum`: additional 50/50 warmup blocks after the required first 50/50 block; can be `0`.
- `BlockLength`: nominal trials per block.
- `BlockMargin`: random plus/minus range around `BlockLength`.
- `BlockEdgeTrials`: first/last trials in majority blocks forced to the majority trial type.
- `MostFraction`: majority trial fraction in short-majority or long-majority blocks.

### Stimulus

- `StimulusMode`: visual only, audio only, or audio + visual.
- `UseSavedImage`: use `image.png` instead of the generated grating.
- `GratingDuration_s`: duration of each stimulus pulse.
- `AudioStimFreq_Hz`: tone frequency.
- `AudioStimVolume`: tone amplitude from 0 to 1.
- `AudioSamplingRate_Hz`: HiFi sample rate.
- `AudioAttenuation_dB`: HiFi attenuation.
- `AudioRamp_ms`: tone onset/offset ramp.

### ISI

- `ShortISIMode`, `LongISIMode`: fixed or uniform random draw.
- `ShortISIFixed_s`, `LongISIFixed_s`: fixed interval values.
- `ShortISIMin_s`, `ShortISIMax_s`, `LongISIMin_s`, `LongISIMax_s`: random draw bounds.

### Manipulation

- `OptoMode`: no opto, random, early trials every block, or early trials in alternating block groups.
- `OptoFraction`: random-mode fraction of eligible opto trials.
- `OptoZeroEdgeTrials`: random-mode first/last trials per block excluded from opto.
- `OptoEarlyTrials`: number of early trials tagged opto in early-trial modes.
- `OptoTriggerType`: Doric trigger type to check before starting the session.
- `OptoTriggerMode`: Doric trigger mode to check before starting the session.
- `EnableOptoStimulus`: if checked, selected opto trials turn on `PWM1` from `AudStimTrigger` onset through spout-in offset.
- `EnableOptoSpoutInDelay`: if checked, selected opto trials turn on `PWM1` during `SpoutInDelay`.
- `EnableOptoChoice`: if checked, selected opto trials turn on `PWM1` during `ChoiceWindow`.
- `EnableOptoPreOutcome`: if checked, selected opto trials turn on `PWM1` during the pre-outcome delay before reward or punish servo-out.
- `EnableOptoPostReward`: if checked, selected opto trials turn on `PWM1` during `PostRewardDelay`.
- `EnableOptoPunishITI`: if checked, selected opto trials turn on `PWM1` during `PunishITI`.
- `ChemoMode`: session-level chemo tag.

Opto settings are synced at the start of each trial, so mid-session changes affect later trials. See [docs/opto-control.md](docs/opto-control.md) for the full opto workflow, timing, plots, and saved fields.

### Probe

- `ProbeMode`: enable probe trial tags.
- `ProbeFraction`: fraction of eligible probe trials.
- `ProbeZeroEdgeTrials`: first/last trials per block excluded from probes.

### Choice, Reward, Servo, ITI

- `SpoutInDelay_s`: delay after stimulus before servo-in command.
- `ChoiceWindow_s`: time allowed for choice.
- `AllowChangeMind`: allow wrong-then-correct rescue.
- `ChangeMindWindow_s`: rescue window duration.
- `PreOutcomeDelay_s`, `PostRewardDelay_s`: outcome and reward timing.
- `LeftRewardAmount_uL`, `RightRewardAmount_uL`: reward sizes.
- `RightServoInPos`, `LeftServoInPos`, `ServoDeflection`, `ServoVelocity`: servo calibration.
- `ServoMoveDelay_s`: time between servo-in command and choice start.
- `ServoReturnTimeout_s`: servo-out timeout.
- `ITIMode`, `PunishITIMode`: manual or bounded exponential.
- `ManualITI_s`, `ManualPunishITI_s`: manual values.
- `ITIMin_s`, `ITIMax_s`, `ITIMean_s`, `PunishITIMin_s`, `PunishITIMax_s`, `PunishITIMean_s`: exponential draw controls.

## Files

- `block_single_interval_discrimination_202606.m`: main protocol.
- `ConfigureProtocol.m`: GUI defaults, metadata, and panel layout.
- `GenerateTrials.m`: trial and block generation.
- `GenerateProbeTrials.m`: probe trial tags.
- `GenerateOptoTrials.m`: opto trial tag allocation.
- `GenerateOptoTrial.m`: current-trial opto assignment from the current GUI settings.
- `OptoControl.m`: opto global timers and plot display metadata.
- `BuildStimulus.m`: visual/audio stimulus generation.
- `BuildStateMachine.m`: Bpod state machine.
- `SoftCodeHandler_BlockSingleInterval.m`: video and servo soft-code actions.
- `ProtocolPlot.m`: live session canvas. The opto plot is filled trial by trial, and the LED1 event row is reconstructed from saved opto periods and state timing.
- `PololuMaestro.m`: servo controller wrapper.

See `docs/` for detailed documentation.

## State Machine Workflow

Every trial starts with `Start`, `VisStimTrigger`, `AudStimTrigger`, and `StimulusDone`. `Start` resets opto timers and raises `BNC1`; `VisStimTrigger` starts the visual stimulus; `AudStimTrigger` plays the audio stimulus when audio is enabled; `StimulusDone` returns the screen and HiFi to grey/off.

Normal trained trials then go through `SpoutInDelay`, `SpoutIn`, and `ChoiceWindow`. A correct lick enters `PreOutcomeDelay`, then `Reward`, then `PostRewardDelay`, then `ServoOut`, then `ITI`. A wrong lick goes to `PreOutcomeDelayPunish`, then `ServoOutPunish`, then `PunishITI`, then `ITI`. If `AllowChangeMind` is on, a wrong lick enters `ChangeMindWindow`; a correct lick there follows the reward path, and timeout follows the punish path. Timeout in `ChoiceWindow` also follows the punish path.

Naive trials use `SpoutInDelay`, `SpoutIn`, `NaiveReward`, and `WaitForCorrectLick`. A correct lick during `WaitForCorrectLick` enters `PostRewardDelay`, then `ServoOut`, then `ITI`; timeout moves the spouts out and enters `ITI`.

Stimulus-only probes exit from `StimulusDone` directly to `ITI`. Servo-only probes go from `StimulusDone` to `ProbeSpoutIn`, then `ProbeChoiceWindow`, then `ServoOut`, then `ITI`; they do not reward or punish choices.

Opto periods are selected independently on trials chosen by `OptoMode`. `EnableOptoStimulus` starts at `AudStimTrigger` onset and stops at the first applicable stimulus/spout-in offset: `SpoutIn` offset on normal trials, `ProbeSpoutIn` offset on servo-only probes, or `AudStimTrigger` offset on stimulus-only probes. If `EnableOptoSpoutInDelay` is also selected, stimulus opto stops at `SpoutInDelay` onset and spout-delay opto covers `SpoutInDelay`. `EnableOptoChoice` starts at `ChoiceWindow` or `ProbeChoiceWindow` onset and stops when that state exits. `EnableOptoPreOutcome` starts at `PreOutcomeDelay` or `PreOutcomeDelayPunish` onset and stops when that state exits. `EnableOptoPostReward` starts at `PostRewardDelay` onset and stops at `PostRewardDelay` offset. `EnableOptoPunishITI` starts at `PunishITI` onset and stops at `PunishITI` offset. Naive sessions save opto tags but force `PWM1` off.
