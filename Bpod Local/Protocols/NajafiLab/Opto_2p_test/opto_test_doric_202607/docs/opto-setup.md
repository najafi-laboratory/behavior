# Opto Setup and Trigger Timing

This protocol is intentionally simple: Bpod controls only when Doric is allowed to stimulate. Doric controls the laser waveform details.

## Trial Timeline

```text
PreStimDelay -> Image1Display -> ImageInterval -> Image2Display -> PostStimDelay -> ITI
```

Opto can be enabled for any combination of four opto epochs:

- pre-stim delay
- image interval
- image 2
- post-stim delay

The ITI is never an opto epoch.

If pre-stim or post-stim opto is enabled, the corresponding delay must be positive. This avoids a zero-length state briefly raising the laser control line at a state boundary.

## Bpod Output

The state machine uses `PWM1` as the opto control line.

- `PWM1 = 255`: Doric input is high.
- `PWM1 = 0`: Doric input is low.

The protocol turns `PWM1` off at trial start and again during ITI as a safety reset.

## Opto Fraction

`OptoFraction` is the probability that a trial is assigned opto.

Examples:

- `OptoFraction = 0`: no opto trials.
- `OptoFraction = 0.2`: about 20 of 100 trials are opto.
- `OptoFraction = 1`: every trial is opto.

For each opto trial, the selected epoch checkboxes decide where opto occurs. If pre-stim, interval, image 2, and post-stim are enabled, then an assigned opto trial receives opto in all four places. `EnableOptoStim` means image 2 only.

## Trigger Mode 1: Full Epoch Gate

GUI setting:

```text
LaserTriggerMode = Full epoch gate
```

Bpod holds `PWM1` high for the whole selected epoch.

| Epoch | GUI duration | Bpod high duration |
| --- | ---: | ---: |
| PreStimDelay | `PreStimDelay_s` | full `PreStimDelay_s` |
| Interval | `ImageInterval_s` | full gray interval |
| Image 2 | `ImageDuration_s` | full second image |
| PostStimDelay | `PostStimDelay_s` | full `PostStimDelay_s` |

Example:

```text
PreStimDelay_s = 1.0
ImageDuration_s = 0.5
ImageInterval_s = 0.5
PostStimDelay_s = 1.5
LaserTriggerMode = Full epoch gate
EnableOptoInterval = off
EnableOptoStim = on
```

On an opto trial, Bpod sends `PWM1` high during image 2.

Best use: Doric is set to generate a pulse train while its external gate is high. Bpod controls the epoch boundaries; Doric controls the pulses.

## Trigger Mode 2: Fixed Duration From Onset

GUI setting:

```text
LaserTriggerMode = Fixed duration from onset
LaserDuration_s = desired Bpod gate length
```

Bpod starts a global timer at the selected epoch onset. The timer drives `PWM1` high for `LaserDuration_s`, then returns it low.

The actual high duration is:

```text
min(LaserDuration_s, selected epoch duration)
```

Examples:

| Epoch | Epoch duration | LaserDuration_s | Actual Bpod high duration |
| --- | ---: | ---: | ---: |
| PreStimDelay | 1.0 s | 0.2 s | 0.2 s |
| Stim | 1.5 s | 0.8 s | 0.8 s |
| PostStimDelay | 2.0 s | 1.0 s | 1.0 s |

Best use: you want a fixed stimulation gate from the start of an epoch, regardless of how long that epoch is.

## Doric Frequency, Time ON, Period, Duty Cycle, And Pulses

The Doric manual describes square pulse sequence fields for frequency/period, time ON/duty cycle, and pulses per sequence. The GUI fields below are reminders for Doric setup:

- `OptoFrequency_Hz`
- `OptoPulseOn_ms`

They are printed before the session starts so the operator can compare MATLAB/Bpod settings with Doric Neuroscience Studio.

This protocol does not synthesize pulse trains in Bpod. It sends one high/low control signal. If you need 50 Hz, 10 ms pulses, configure that waveform in Doric and use Bpod `PWM1` as the external gate.

Use these conversions when setting up Doric:

- `Frequency`: pulses per second in Hz.
- `Period`: time between pulse starts in ms. `Period_ms = 1000 / Frequency_Hz`.
- `Time ON`: duration of one pulse in ms. Match this to `OptoPulseOn_ms`.
- `Duty Cycle`: percent of one period that is ON. `DutyCycle_% = 100 * TimeON_ms / Period_ms`.
- `Pulse(s) per Sequence`: number of pulses in one Doric sequence. For external gated operation, set this to `0` for infinite pulses and let Bpod's `PWM1` gate define the train length. For a standalone fixed sequence, use `ceil(GateDuration_s * Frequency_Hz)`.

Default example:

```text
OptoFrequency_Hz = 50
OptoPulseOn_ms = 10
Period_ms = 1000 / 50 = 20
DutyCycle_% = 100 * 10 / 20 = 50
Pulse(s) per Sequence = 0 for Bpod-gated operation
```

If a standalone Doric sequence should last 0.5 s at 50 Hz, set `Pulse(s) per Sequence` to about `ceil(0.5 * 50) = 25`.

## Concrete Setup Examples

### 50 Hz Pulses During Image 2

Bpod GUI:

```text
OptoMode = on
OptoFraction = 0.5
EnableOptoPreStimDelay = off
EnableOptoInterval = off
EnableOptoStim = on
EnableOptoPostStimDelay = off
LaserTriggerMode = Full epoch gate
ImageDuration_s = 0.5
ImageInterval_s = 0.5
OptoFrequency_Hz = 50
OptoPulseOn_ms = 10
```

Doric:

```text
External gate enabled
Pulse frequency = 50 Hz
Pulse width = 10 ms
```

Result: about half the trials produce a 0.5 s Doric pulse train during image 2.

### 100 ms Laser at Stimulus Onset

Bpod GUI:

```text
OptoMode = on
OptoFraction = 1
EnableOptoStim = on
LaserTriggerMode = Fixed duration from onset
LaserDuration_s = 0.1
ImageDuration_s = 0.5
```

Result: every trial produces a 100 ms Bpod gate beginning at image 2 onset.

### Full Pre-Stim Plus Short Post-Stim

This protocol uses one `LaserTriggerMode` for all enabled epochs. To run full pre-stim delay opto and short post-stim onset opto in the same session, use one of these approaches:

- Run one session with `Full epoch gate` for pre-stim-only opto, then a second session with `Fixed duration from onset` for post-stim-only opto.
- Keep both epochs enabled with `Fixed duration from onset` if the same fixed duration is acceptable for both epochs.

## Saved Data

`BpodSystem.Data.OptoTrialTypes` is a `4 x nTrials` matrix.

Rows:

1. pre-stim delay
2. image interval
3. image 2
4. post-stim delay

Values:

- `0`: opto off for that epoch on that trial
- `1`: opto on for that epoch on that trial

`BpodSystem.Data.ITI` stores the ITI duration used for each completed trial.
