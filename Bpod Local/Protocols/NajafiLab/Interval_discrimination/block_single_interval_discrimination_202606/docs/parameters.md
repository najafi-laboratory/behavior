# Parameters

## Session

- `MaxTrials`: maximum trials in the session.
- `TrainingMode`: `Naive` or `Trained`.
- `Contingency`: target-side mapping.
  - `Short-left, long-right`
  - `Short-right, long-left`
- `BlockNum`: block construction mode.
  - `50/50 only`
  - `50/50 then left/right`
  - `50/50, left, right`
- `WarmupBlockNum`: additional 50/50 warmup blocks after the required first 50/50 block in block modes 2 and 3. Set to `0` to use only the required first 50/50 block before the configured block structure.
- `BlockLength`: nominal block length.
- `BlockMargin`: random plus/minus range around `BlockLength`.
- `BlockEdgeTrials`: edge trials forced toward the block's majority trial type.
- `MostFraction`: majority fraction in short-majority or long-majority blocks.

## Stimulus

- `StimulusMode`: visual only, audio only, or audio + visual.
- `UseSavedImage`: when checked, use `image.png` in the protocol folder as the visual stimulus image.
- `GratingDuration_s`: duration of each visual stimulus pulse.
- `AudioStimFreq_Hz`: tone frequency.
- `AudioStimVolume`: tone amplitude, 0 to 1.
- `AudioSamplingRate_Hz`: HiFi sample rate.
- `AudioAttenuation_dB`: HiFi attenuation.
- `AudioRamp_ms`: tone onset/offset ramp.

## ISI

Short and long trial types each have fixed or uniform random ISI control:

- `ShortISIMode`
- `ShortISIFixed_s`
- `ShortISIMin_s`
- `ShortISIMax_s`
- `LongISIMode`
- `LongISIFixed_s`
- `LongISIMin_s`
- `LongISIMax_s`

## Manipulation

- `OptoMode`: controls which trials are selected as opto trials. It does not by itself choose the opto period; period checkboxes below decide which epochs receive light on selected opto trials.
  - `No opto`: every trial is opto off. Period checkboxes are ignored.
  - `Random`: each eligible trial is independently sampled when the trial starts. The probability is `OptoFraction`. Warmup blocks and block-edge trials are excluded.
  - `Early trials in every block`: after warmup blocks, the first `OptoEarlyTrials` trials of every block are opto trials.
  - `Early trials in alternating block groups`: after warmup blocks, the first `BlockNum` blocks are no-opto, the next `BlockNum` blocks have early opto trials, and this no-opto/opto group pattern repeats.
- `OptoFraction`: probability for an eligible trial to become opto in `Random` mode. Example: `0.35` means each eligible trial has a 35% chance when it starts. This value is ignored in early-trial modes.
- `OptoZeroEdgeTrials`: number of trials at the start and end of each non-warmup block forced opto off in `Random` mode. This protects block transitions from opto contamination. Warmup blocks are fully forced off regardless of this value.
- `OptoEarlyTrials`: number of early trials per eligible block selected in the two early-trial modes. If the value exceeds the block length, the whole block can be tagged. This value is ignored in `Random` and `No opto` modes.
- `EnableOptoStimulus`: when checked, selected opto trials drive `PWM1` high from `AudStimTrigger` onset through spout-in offset. This period can span multiple Bpod states.
- `EnableOptoChoice`: when checked, selected opto trials drive `PWM1` high during the actual `ChoiceWindow`. If the animal licks early and exits the choice state, the choice opto timer is cancelled.
- `EnableOptoPreReward`: when checked, selected opto trials drive `PWM1` high from `PreRewardDelay` onset through `Reward` offset.
- `EnableOptoPostReward`: when checked, selected opto trials drive `PWM1` high during `PostRewardDelay`.
- `EnableOptoPunishITI`: when checked, selected opto trials drive `PWM1` high during `PunishITI`.
- `ChemoMode`: session-level chemo flag saved to all completed trials once enabled.

Selected opto trials can enable any combination of the five periods. In trained sessions, enabled periods drive `PWM1` high through global timers. In naive sessions, opto output is forced off even if tags exist.

Opto settings are synced at the start of every trial. Changing `OptoMode`, `OptoFraction`, `OptoEarlyTrials`, or any period checkbox during a session affects the next trial that has not yet started. Completed trials are not rewritten.

## Probe

- `ProbeMode`: enables probe trial tagging.
- `ProbeFraction`: fraction of eligible trials tagged probe.
- `ProbeZeroEdgeTrials`: trials near block edges forced probe-off.

Probe type `1` is stimulus only. Probe type `2` moves the spouts in for the choice-window duration and then moves them out.

## Choice

- `SpoutInDelay_s`: delay after stimulus before servo-in command.
- `ChoiceWindow_s`: choice window duration.
- `AllowChangeMind`: enables correction after first wrong lick.
- `ChangeMindWindow_s`: duration of the correction window.

## Reward

- `PreRewardDelay_s`: delay between correct lick and valve opening.
- `PostRewardDelay_s`: delay after reward before servo out.
- `LeftRewardAmount_uL`
- `RightRewardAmount_uL`

## Servo

- `RightServoInPos`
- `LeftServoInPos`
- `ServoDeflection`
- `ServoVelocity`
- `ServoMoveDelay_s`: duration reserved for servo movement before choice starts.
- `ServoReturnTimeout_s`: timeout for servo-out acknowledgement.

## ITI

- `ITIMode`: manual or exponential.
- `ManualITI_s`
- `ITIMin_s`
- `ITIMax_s`
- `ITIMean_s`
- `PunishITIMode`: manual or exponential.
- `ManualPunishITI_s`
- `PunishITIMin_s`
- `PunishITIMax_s`
- `PunishITIMean_s`
