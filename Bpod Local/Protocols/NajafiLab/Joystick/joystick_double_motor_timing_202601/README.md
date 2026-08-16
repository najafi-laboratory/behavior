# Joystick Double Motor Timing Protocol

This is a compact Bpod MATLAB protocol for a joystick timing task with configurable sensory cues, servo control, rotary encoder thresholds, optional assist trials, opto/probe/chemo trial tags, and online plotting.

## Update note

### 2026.06.24
- First release.

### 2026.06.30
- Added opto during pre reward delay.
- Changed reward delay namings.
- Adjusted canvas layouts.
- Added BNC2 to event plot.

### 2026.07.22
- Added sensory cue mode: visual only, audio only, or audio + visual.
- Renamed cue-facing GUI fields from visual cue wording to sensory cue wording.
- Changed event plot lick row from port 1 to port 2.
- Added 8 digit session date to the first summary line.

### 2026.07.27
- Turn off the Bpod console status LED when the protocol starts.
- Prefer HiFi audio and fall back to the current system speaker.
- Preload synchronized audio and a dedicated gray idle display frame.
- Modularize audio and idle-display handling.

### 2026.08.06
- Added reward impulse.
- Changed default parameters.

### 2026.08.12
- Apply the normal inter-trial idle-screen configuration while waiting for the session-start Enter, matching the screen setup immediately before the first real trial.

### 2026.08.14
- Route the pre-session idle screen through `SoftCodeHandler_Protocol(3)`, matching the pre-trial setup and keeping the configured display visible when another window is clicked.

## Main Workflow

1. Run `joystick_double_motor_timing_202601`.
2. The Bpod console status LED turns off, then the GUI opens. Set parameters and press Enter in MATLAB.
3. Hardware is configured: Pololu Maestro servo, rotary encoder, HiFi or system-speaker audio, and PsychToolbox video display.
4. The servo returns home and `SoftCodeHandler_Protocol(3)` applies the normal inter-trial idle screen. Press Enter at the ready prompt to start trials.
5. Each trial syncs GUI parameters, builds the next state machine, runs Bpod, saves trial data, and updates the plot canvas.

## Trial Logic

The state machine starts with `SensoryCue1`, then either waits for `Press1` in double-press mode or goes directly to press 2 in single-press mode.

For press 2, timing is measured from the start of the press 2 window. The state `Press2` captures the timing and `RewardLeverRetract` routes the trial to `EarlyPress2`, `PreRewardDelay`, or `Press2Late`. Missing presses go to `DidNotPress1` or `DidNotPress2`. Rewarded trials finish through `Reward`, `PostRewardDelay`, and `LeverRetractFinal`.

Assist trials hold the servo until perfect timing, release it, and then use the same `Press2` outcome path.

Probe type 1 omits water reward even for rewarded timing. Probe type 2 flips the timing mode for that trial: visual guided becomes self timed, and self timed becomes visual guided.

Opto and probe tags are forced to zero for the first block and for the first/last trials of each block. The edge lengths are controlled separately by `OptoZeroEdgeTrials` and `ProbeZeroEdgeTrials`. Probe trials are excluded from opto tagging.

Opto sessions require `AssistMode` off. The protocol prints this check before the session and stops if opto and assist are both enabled.

Opto trials are saved as a `4 x nTrials` matrix, with rows for sensory cue 1, delay, pre reward delay, and post reward. A selected trial can enable any combination of these periods. LED1 turns on during `SensoryCue1` for cue opto, from `LeverRetract1` to `RewardLeverRetract` for delay opto, during `PreRewardDelay` until `Reward` for pre-reward-delay opto, and during `PostRewardDelay` until `LeverRetractFinal` for post-reward opto.

`ChemoMode` stores `ChemoTrialTypes` as 1 when enabled and 0 otherwise.

## Reward Shape

Reward amount is computed in `SoftCodeHandler_Protocol` from the press 2 time:

- zero at `perfect timing - RewardWindowLeft_s`
- maximum at perfect timing
- maximum through `RewardMaximumWindow_s`
- linearly decreases to zero over `RewardWindowRight_s`

The resulting amount is converted to calibrated valve-on time for valve 2. The
protocol divides `TotalRewardDuration_s` into exactly 10 hard-coded, identical
square-wave cycles. Each cycle contains a valve-on period of `valve time / 10`
followed by a valve-off period of `(total reward duration - valve time) / 10`.
Thus the duty cycle is `valve time / total reward duration`, the full calibrated
valve-on time is preserved, and the impulses are uniformly distributed. The
configured duration must be longer than the calibrated valve time.

## GUI Parameters

### Session

- `MaxTrials`: maximum trials before the protocol stops.
- `PressMode`: single press or double press task.
- `TrialMode`: all short, all long, short-first blocks, or long-first blocks.
- `BlockLength`: nominal number of trials per short/long block.
- `BlockLengthEdge`: random block-edge jitter added to block length.
- `ProbeMode`: enables probe trial tagging.
- `ProbeFraction`: fraction of eligible trials tagged as probes.
- `ProbeZeroEdgeTrials`: block-edge trials excluded from probe tagging.

### Stimulus

- `TimingMode`: visual guided or self timed.
- `SensoryCueMode`: visual only, audio only, or audio + visual.
- `SensoryCueDuration_s`: duration of sensory cue 1 and cue 2.
- `UseGeneratedGrating`: use generated grating instead of `image.png`.

### Audio

- `AudioStimFreq_Hz`: sensory cue tone frequency.
- `AudioStimVolume`: sensory cue tone amplitude from 0 to 1.
- `AudioSamplingRate_Hz`: Sampling rate used by HiFi and system-speaker audio.
- `AudioAttenuation_dB`: HiFi digital attenuation.
- `AudioRamp_ms`: onset and offset ramp for the tone.

The protocol tries HiFi first, then the current system speaker through a pre-opened PsychPortAudio stream. Audio is buffered before each trial and triggered immediately after the display flip that turns the sync patch light. If neither output is available, the protocol prints a reminder and continues without auditory cue output.

During trials, the display stays gray with the patch dark except while a cue frame is shown. Static cue and idle textures use a non-waiting Psychtoolbox flip so a stalled vertical-blank synchronization cannot block Bpod in `SensoryCue1`. Changing `SensoryCueMode` during a session updates the trial display states before the incoming trial. Duplicate soft-code requests for a visual state that is already displayed are ignored.

### Timing

- `ShortDelay_s`: target press 2 delay for short trials.
- `LongDelay_s`: target press 2 delay for long trials.
- `Press1Window_s`: time allowed for press 1.
- `ShortPress2Window_s`: press 2 response window on short trials.
- `LongPress2Window_s`: press 2 response window on long trials.

### Joystick

- `PressThreshold`: encoder threshold for a joystick press.
- `RetractThreshold`: encoder threshold used for retract/home logic.
- `ServoInPos`: servo home position.
- `ServoOutPos`: servo released position.
- `ServoMoveDelay_s`: wait after detected press before retract routing.
- `LeverRetract1Duration_s`: fixed wait after starting the first lever retraction (default 0.1 s).
- `ServoReturnTimeout_s`: maximum wait for servo return soft-code confirmation.
- `AssistMode`: enables assist trials after early press 2.
- `AssistFraction`: probability of assist after an eligible early trial.

### Reward

- `RewardWindowLeft_s`: early side of rewarded timing window.
- `RewardMaximumWindow_s`: plateau after perfect timing with maximum reward.
- `RewardWindowRight_s`: late side of rewarded timing window.
- `PreRewardDelay_s`: delay from rewarded classification to water delivery.
- `PostRewardDelay_s`: delay after reward before final rewarded state.
- `TotalRewardDuration_s`: window containing 10 identical reward cycles; defaults to 2 seconds.
- `RewardMode`: same reward for all trials or separate short/long reward sizes.
- `RewardAmount_uL`: reward size when using same reward mode.
- `ShortRewardAmount_uL`: short-trial reward size in different reward mode.
- `LongRewardAmount_uL`: long-trial reward size in different reward mode.

### ITI

- `ITIMode`: manual or truncated exponential normal ITI.
- `ManualITI_s`: normal ITI when manual mode is selected.
- `ITIMin_s`, `ITIMax_s`, `ITIMean_s`: exponential normal ITI bounds and mean.
- `PunishITIMode`: manual or truncated exponential punish ITI.
- `ManualPunishITI_s`: punish ITI when manual mode is selected.
- `PunishITIMin_s`, `PunishITIMax_s`, `PunishITIMean_s`: exponential punish ITI bounds and mean.

### Manipulation

- `OptoMode`: enables opto trial tagging.
- `OptoFraction`: fraction of eligible trials tagged as opto.
- `OptoZeroEdgeTrials`: block-edge trials excluded from opto tagging.
- `EnableOptoSensoryCue1`: adds cue 1 light to selected opto trials.
- `EnableOptoDelay`: adds delay-period light to selected opto trials.
- `EnableOptoPreRewardDelay`: adds pre-reward-delay light to selected opto trials.
- `EnableOptoPostReward`: adds post-reward light to selected opto trials.
- `OptoFrequency_Hz`: reserved pulse frequency setting; current output is sustained high.
- `OptoPulseOn_ms`: reserved pulse on-time setting; current output is sustained high.
- `ChemoMode`: marks completed trials as chemo trials in saved data.

## Main Files

- `joystick_double_motor_timing_202601.m`: main protocol, GUI sync, hardware setup, trial loop, data saving.
- `ConfigureProtocol.m`: GUI defaults, metadata, and parameter panels.
- `BuildStateMachine.m`: Bpod state machine for each trial.
- `SoftCodeHandler_Protocol.m`: servo, sensory cue, press timing, and reward delivery soft-code operations.
- `GenerateTrials.m`: short/long trial blocks plus ITI and punish ITI generation.
- `OptoControl.m`: opto trial tags and neutral display colors.
- `ProbeControl.m`: probe trial tags and neutral display colors.
- `GenerateSensoryCueVideo.m`: creates the visual part of cue frames from `image.png` or the generated sinusoidal grating.
- `ProtocolPlot.m`: one online canvas for trial outcomes, opto/probe tags, delay, press timing, encoder, states, and BNC/lick events.

## Plot Notes

The online plot uses a two-column canvas: the left side shows trial type, opto period, probe type, delay, and rotary encoder plots; the right side shows outcome fractions with a legend, press timing, state timing, and events. Outcome-related plots share one color set. Opto, probe, delay, and event plots use neutral gray or black marks. The rotary encoder position trace is black, while its threshold and event markers use colored annotations.

The event plot uses one row per signal: `BNC 1`, `BNC 2`, `LED 1`, and `Port 2 lick`. Filled bars mark logical 1 or detected lick pulses; blank space marks 0.
