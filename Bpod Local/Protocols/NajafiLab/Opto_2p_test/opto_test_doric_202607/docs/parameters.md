# Parameters

All parameters live in `S.GUI`.

## Session

The protocol runs up to 1000 trials. This limit is hard-coded in `opto_test_doric_202607.m` and is not shown in the GUI.

## Timing

### `PreStimDelay_s`

Gray delay before the first image. Must be positive when pre-stim opto is enabled.

### `ImageDuration_s`

Requested duration for each image. The actual value is rounded to a whole number of display frames.

### `ImageInterval_s`

Gray interval between image 1 and image 2.

### `PostStimDelay_s`

Gray delay after image 2. Must be positive when post-stim opto is enabled.

## ITI

### `ITIMode`

- `1`: manual fixed ITI
- `2`: bounded exponential ITI

### `ManualITI_s`

Fixed ITI when manual mode is selected. Default is 1 s.

### `ITIMin_s`, `ITIMax_s`, `ITIMean_s`

Bounds and mean parameter for bounded exponential ITI sampling. Defaults match the reference protocol: min 3 s, max 5 s, mean 4 s.

## Opto

### `OptoMode`

Enables opto assignment.

### `OptoFraction`

Probability that each trial is assigned opto.

### `EnableOptoPreStimDelay`

Adds opto during `PreStimDelay` on assigned opto trials.

### `EnableOptoStim`

Adds opto during `Image2Display` on assigned opto trials.

### `EnableOptoInterval`

Adds opto during `ImageInterval` on assigned opto trials.

### `EnableOptoPostStimDelay`

Adds opto during `PostStimDelay` on assigned opto trials.

### `LaserTriggerMode`

- `1`: full epoch gate
- `2`: fixed duration from epoch onset

### `LaserDuration_s`

Duration of the Bpod gate when `LaserTriggerMode` is fixed duration from onset. The gate is clipped to the selected epoch duration.

### `OptoFrequency_Hz` and `OptoPulseOn_ms`

Operator-facing Doric settings. The Bpod protocol does not synthesize the pulse train; it gates Doric through `PWM1`.
