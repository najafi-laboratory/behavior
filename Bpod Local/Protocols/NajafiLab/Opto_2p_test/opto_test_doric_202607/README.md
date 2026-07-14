# Opto Test Doric 202607

Minimal Bpod protocol for Doric opto timing tests with one generated grating stimulus shown twice per trial. If a root `image.png` is present, the protocol uses it; otherwise it generates the grating in MATLAB.

Each trial is:

1. `PreStimDelay`
2. `Image1Display`
3. `ImageInterval`
4. `Image2Display`
5. `PostStimDelay`
6. `ITI`

There are no responses, rewards, probe trials, or servo/encoder dependencies. The grating is generated in MATLAB using the same sinusoidal frame parameters as the `ref` protocol.

Details on Doric Neuroscience Studio is available in the official manual:

https://doriclenses.com/downloads/UserManual/Doric_Neuroscience_Studio-MANUALv5.3.4.pdf

## Update note

### 2026.07.09
- First release.

### 2026.07.14
- Added image-interval-image trial states.
- Updated the session canvas to show opto trial types, state timing, and LED1 event timing.

## Run

Run this from the Bpod protocol folder:

```matlab
opto_test_doric_202607
```

The protocol opens the Bpod parameter GUI, waits for you to edit settings, configures the PsychToolbox display, shows gray, then waits for Enter to start the session.

## Main GUI Parameters

- `PreStimDelay_s`: delay before image onset.
- `ImageDuration_s`: grating display duration for each image. The protocol snaps this to the nearest video frame.
- `ImageInterval_s`: gray interval between the two grating displays.
- `PostStimDelay_s`: delay after image offset.
- `ITIMode`: manual or exponential ITI.
- `ManualITI_s`: fixed ITI when manual mode is selected.
- `ITIMin_s`, `ITIMax_s`, `ITIMean_s`: bounded exponential ITI settings.
- `OptoMode`: enables opto trial assignment.
- `OptoFraction`: fraction of trials assigned as opto trials.
- `EnableOptoPreStimDelay`: opto can occur during `PreStimDelay`.
- `EnableOptoInterval`: opto can occur during `ImageInterval`.
- `EnableOptoStim`: opto can occur during `Image2Display`.
- `EnableOptoPostStimDelay`: opto can occur during `PostStimDelay`.
- `LaserTriggerMode`: chooses how Bpod drives Doric on selected opto epochs.
- `LaserDuration_s`: fixed Bpod gate duration when `LaserTriggerMode` is fixed duration.
- `OptoFrequency_Hz`, `OptoPulseOn_ms`: Doric settings printed for the operator to verify.

`PreStimDelay_s` and `PostStimDelay_s` may be zero only when their opto checkboxes are off. If opto is enabled for one of those epochs, that epoch must have a positive duration.

## Opto Trial Assignment

The protocol runs up to 1000 trials. This limit is hard-coded in `opto_test_doric_202607.m` and is not a GUI parameter.

When `OptoMode` is off, every trial receives `[0; 0; 0; 0]`.

When `OptoMode` is on, each trial is independently assigned opto with probability `OptoFraction`. On assigned trials, all enabled opto epochs are active. For example:

- `OptoFraction = 0.25`, only `EnableOptoStim = on`: about 25% of trials have opto during image 2.
- `OptoFraction = 0.50`, `EnableOptoPreStimDelay = on` and `EnableOptoPostStimDelay = on`: about half of trials have opto in both the pre-stim and post-stim delays.
- `OptoFraction = 1`, all four epoch checkboxes on: every trial has opto in all four epochs.

Completed assignments are saved in `BpodSystem.Data.OptoTrialTypes` as a `4 x nTrials` matrix:

1. row 1: pre-stim delay opto
2. row 2: image interval opto
3. row 3: image 2 opto
4. row 4: post-stim delay opto

## Trigger Options and Laser Durations

The protocol drives Bpod `PWM1` / LED1. Doric should be configured to accept that line as the external gate or trigger input.

### Full Epoch Gate

Set `LaserTriggerMode = Full epoch gate`.

Bpod sets `PWM1` high at the start of each selected epoch and turns it off at the next state transition. The Bpod laser gate duration is therefore the full epoch duration.

Examples:

- `PreStimDelay_s = 1.000`, pre-stim opto on: `PWM1` is high for 1.000 s during the pre-stim delay.
- `ImageInterval_s = 0.500`, interval opto on: `PWM1` is high for 0.500 s during the gray interval.
- `ImageDuration_s = 0.500`, stim opto on: `PWM1` is high for 0.500 s during image 2.
- `PostStimDelay_s = 2.000`, post-stim opto on: `PWM1` is high for 2.000 s during the post-stim delay.

Use this mode when Doric is configured as a gated pulse train generator. In that setup, Bpod controls when the laser train is allowed to run, while Doric controls the pulse frequency and pulse width.

### Fixed Duration From Onset

Set `LaserTriggerMode = Fixed duration from onset`.

Bpod starts a global timer at the beginning of each selected epoch. That timer sets `PWM1` high for `LaserDuration_s`, then returns it low. If `LaserDuration_s` is longer than the epoch, the protocol clips the timer to the epoch duration.

Examples:

- `PreStimDelay_s = 1.000`, `LaserDuration_s = 0.250`, pre-stim opto on: `PWM1` is high for the first 0.250 s of pre-stim delay, then off for the remaining 0.750 s.
- `ImageDuration_s = 0.500`, `LaserDuration_s = 0.100`, stim opto on: `PWM1` is high for the first 0.100 s of image 2.
- `PostStimDelay_s = 0.300`, `LaserDuration_s = 1.000`, post-stim opto on: `PWM1` is clipped to 0.300 s because the selected epoch ends after 0.300 s.

Use this mode when you want the laser gate duration to be independent of the behavioral epoch length.

## Doric Pulse Setup

Use the Doric manual's square sequence fields when configuring the Doric pulse generator:

- `Frequency`: pulses per second in Hz.
- `Period`: time between pulse starts in ms. `Period_ms = 1000 / Frequency_Hz`.
- `Time ON`: duration of one pulse in ms. Match this to `OptoPulseOn_ms`.
- `Duty Cycle`: percent of each period that is ON. `DutyCycle_% = 100 * TimeON_ms / Period_ms`.
- `Pulse(s) per Sequence`: number of pulses in one sequence. In gated use, set this to `0` for infinite pulses so Bpod controls the train length by holding `PWM1` high. For fixed-duration sequences, use `ceil(GateDuration_s * Frequency_Hz)`.

For the default `OptoFrequency_Hz = 50` and `OptoPulseOn_ms = 10`:

- `Period = 1000 / 50 = 20 ms`.
- `Duty Cycle = 100 * 10 / 20 = 50%`.
- If Bpod gates image 2 for `ImageDuration_s = 0.5`, use infinite pulses (`0`) for external gated operation, or about `25` pulses for a standalone fixed sequence.

## Doric Setup Checklist

1. Connect Bpod `PWM1` / LED1 output to the Doric external TTL/gate input used for stimulation.
2. In Doric Neuroscience Studio, configure the light source or pulse generator to use Bpod as the external gate/trigger.
3. In the Doric square sequence settings, set frequency or period, time ON or duty cycle, and pulses per sequence using the formulas above.
4. If using `Full epoch gate`, set Doric to produce pulses while the external input is high.
5. If using `Fixed duration from onset`, Doric still sees a high gate from Bpod, but Bpod limits that gate to `LaserDuration_s`.
6. Before starting, confirm in MATLAB that the printed opto settings match Doric.

Important distinction: `OptoFrequency_Hz` and `OptoPulseOn_ms` do not create a pulse train inside this Bpod protocol. They are operator-facing Doric settings. Bpod only controls the timing of the external `PWM1` gate.

## Files

- `opto_test_doric_202607.m`: main protocol loop and hardware setup.
- `ConfigureProtocol.m`: GUI defaults and panel metadata.
- `BuildStateMachine.m`: pre-stim, image-interval-image, post-stim, and ITI states.
- `GenerateTrials.m`: reference-style manual/exponential ITI generation.
- `GenerateVisualCueVideo.m`: loads optional `image.png` or generates a sinusoidal grating frame.
- `OptoControl.m`: opto trial assignment and `PWM1` actions.
- `SoftCodeHandler_Protocol.m`: starts/stops the PsychToolbox grating video.
- `docs/opto-setup.md`: detailed trigger and laser-duration guide.
