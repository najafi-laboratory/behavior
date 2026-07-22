# Parameters

This page lists the GUI parameters.

All parameters live in `S.GUI`.

## Session

### `MaxTrials`

Total planned trials.

The trial loop stops when this number is reached.

### `PressMode`

Selects the task structure.

- `1`: Single press.
- `2`: Double press.

In double press mode, cue 1 leads to press 1 first.

In single press mode, cue 1 leads directly to press 2 timing.

### `TrialMode`

Controls short and long trial order.

- `1`: All short.
- `2`: All long.
- `3`: Blocks, short first.
- `4`: Blocks, long first.

### `BlockLength`

Mean block length for block modes.

### `BlockLengthEdge`

Random range around `BlockLength`.

If `BlockLength` is 30 and `BlockLengthEdge` is 5, block lengths range from 25 to 35.

### `ProbeMode`

Enables probe trial tagging.

### `ProbeFraction`

Fraction of eligible trials tagged as probe trials.

### `ProbeZeroEdgeTrials`

Number of trials blocked at each block edge.

The first block is also blocked.

Probe type 1 omits reward.

Probe type 2 flips timing mode for that trial.

## Stimulus

### `TimingMode`

Controls cue 2.

- `1`: Visual guided.
- `2`: Self timed.

In visual-guided mode, cue 2 appears at the target time.

In self-timed mode, cue 2 is not shown.

### `SensoryCueMode`

Controls the sensory cue modality.

- `1`: Visual only.
- `2`: Audio only.
- `3`: Audio + visual.

### `SensoryCueDuration_s`

Requested sensory cue duration.

The actual duration is rounded to a whole number of video frames.

### `UseGeneratedGrating`

Controls the visual cue source.

- `1`: Use generated grating.
- `0`: Use `image.png`.

This is ignored for audio-only cues.

## Audio

### `AudioStimFreq_Hz`

Tone frequency for audio-only and audio+visual sensory cues.

If the HiFi module is missing, the protocol prints a warning and continues without auditory cue output.

### `AudioStimVolume`

Tone amplitude from 0 to 1.

### `AudioSamplingRate_Hz`

HiFi module sampling rate.

### `AudioAttenuation_dB`

HiFi digital attenuation.

### `AudioRamp_ms`

Tone onset and offset ramp duration in milliseconds.

## Timing

### `ShortDelay_s`

Target press 2 time for short trials.

### `LongDelay_s`

Target press 2 time for long trials.

### `Press1Window_s`

Time allowed for press 1.

Only used when `PressMode` is double press.

### `ShortPress2Window_s`

Time allowed for press 2 on short trials.

### `LongPress2Window_s`

Time allowed for press 2 on long trials.

Each press 2 window must contain the full reward window.

## Joystick

### `PressThreshold`

Encoder position needed to count as a press.

### `RetractThreshold`

Encoder position used to decide when the lever has returned home.

### `ServoInPos`

Servo position for the home state.

### `ServoOutPos`

Servo offset used to place the lever out.

### `ServoMoveDelay_s`

Delay after a detected press before moving the servo back.

### `ServoReturnTimeout_s`

Maximum time to wait for the lever to return near zero.

### `AssistMode`

Enables assisted trials.

This must be off during opto sessions.

### `AssistFraction`

Probability that an eligible trial becomes assisted.

A trial is eligible only if the previous trial had an early press 2.

## Reward

### `RewardWindowLeft_s`

Time before the target where reward ramps up.

Presses earlier than this are early.

### `RewardMaximumWindow_s`

Flat maximum reward window after the target.

### `RewardWindowRight_s`

Time after the maximum window where reward ramps down.

Presses later than this are late.

### `PreRewardDelay_s`

Delay between a rewarded press 2 outcome and reward delivery.

Pre-reward-delay and post-reward opto do not change this value.

### `PostRewardDelay_s`

Delay after reward before the trial is marked as rewarded.

Post-reward opto gives light during this period.

### `RewardMode`

Controls reward amount.

- `1`: Same reward for short and long.
- `2`: Separate rewards for short and long.

### `RewardAmount_uL`

Maximum reward when `RewardMode` is same reward.

### `ShortRewardAmount_uL`

Maximum reward for short trials when `RewardMode` is different reward.

### `LongRewardAmount_uL`

Maximum reward for long trials when `RewardMode` is different reward.

## ITI

### `ITIMode`

Controls normal ITI.

- `1`: Manual.
- `2`: Exponential.

### `ManualITI_s`

Fixed normal ITI when `ITIMode` is manual.

### `ITIMin_s`

Minimum normal ITI for exponential mode.

### `ITIMax_s`

Maximum normal ITI for exponential mode.

### `ITIMean_s`

Mean parameter for exponential ITI sampling.

The remaining ITI fields control the delay after an error.

### `PunishITIMode`

Controls ITI after an error.

- `1`: Manual.
- `2`: Exponential.

### `ManualPunishITI_s`

Fixed punish ITI when manual mode is used.

### `PunishITIMin_s`

Minimum punish ITI for exponential mode.

### `PunishITIMax_s`

Maximum punish ITI for exponential mode.

### `PunishITIMean_s`

Mean parameter for punish ITI sampling.

## Manipulation

### `OptoMode`

Enables opto trial tagging.

Opto sessions require `AssistMode` off. Probe trials are excluded from opto tagging.

### `OptoFraction`

Fraction of eligible trials tagged as opto trials.

### `OptoZeroEdgeTrials`

Number of trials blocked at each block edge.

The first block is also blocked.

### `EnableOptoSensoryCue1`

Adds cue 1 light to selected opto trials.

This sets row 1 of the saved opto matrix.

### `EnableOptoDelay`

Adds delay-period light to selected opto trials.

This sets row 2 of the saved opto matrix. LED1 runs from `LeverRetract1` until `RewardLeverRetract`, or until the press 2 window ends.

### `EnableOptoPreRewardDelay`

Adds pre-reward-delay light to selected opto trials.

This sets row 3 of the saved opto matrix. LED1 runs during `PreRewardDelay` and turns off in `Reward`.

### `EnableOptoPostReward`

Adds post-reward light to selected opto trials.

This sets row 4 of the saved opto matrix. LED1 runs during `PostRewardDelay` and turns off in `LeverRetractFinal`.

### `OptoFrequency_Hz`

Doric square-wave frequency reserved for pulsed stimulation.

The current state machine keeps LED1 continuously high during the selected opto epoch.

### `OptoPulseOn_ms`

Doric pulse on time reserved for pulsed stimulation.

The current state machine keeps LED1 continuously high during the selected opto epoch.

### `ChemoMode`

Marks chemo status in saved data.

If any completed trial is chemo, all completed chemo tags are set to 1.
