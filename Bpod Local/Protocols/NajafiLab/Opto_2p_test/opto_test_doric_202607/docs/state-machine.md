# State Machine

`BuildStateMachine.m` creates one Bpod state machine per trial.

## States

### `Start`

Turns `BNC1` high, sets the display to gray, turns `PWM1` off, cancels opto timers, then enters `PreStimDelay`.

### `PreStimDelay`

Shows gray for `PreStimDelay_s`. Pre-stim opto starts here when enabled for the trial.

### `Image1Display`

Plays video slot 1 for `ImageDuration_s`.

### `ImageInterval`

Shows gray for `ImageInterval_s`. Interval opto starts here when enabled for the trial.

### `Image2Display`

Plays video slot 2 for `ImageDuration_s`. Stim opto starts here when enabled for the trial.

### `PostStimDelay`

Shows gray for `PostStimDelay_s`. Image 2 opto is turned off at this transition unless post-stim opto is enabled. Post-stim opto starts here when enabled for the trial.

### `ITI`

Shows gray for the assigned ITI, turns `PWM1` off, turns `BNC1` low, and exits.

## Opto Timing

In full epoch gate mode, `PWM1` is held high for the full selected epoch.

In fixed duration mode, global timers 1, 2, and 3 gate `PWM1` for pre-stim, stim, and post-stim epochs. Each timer duration is `min(LaserDuration_s, epoch duration)`.
