# Plots And Data

## Online Canvas

`ProtocolPlot.m` creates one figure at the beginning of the session and updates it after each trial.

The canvas shows:

- opto trial types across the session
- completed-trial state timing from raw Bpod events
- event timing for `BNC1` and reconstructed LED1 / `PWM1`

## Saved Fields

### `BpodSystem.Data.PlannedOptoTrialTypes`

Planned and online-updated opto assignments. This is a `4 x nTrials` matrix.

### `BpodSystem.Data.OptoTrialTypes`

Completed trial opto assignments. This is a `4 x completedTrials` matrix.

### `BpodSystem.Data.AssignedOptoTrialCount`

Highest trial number that has received an online opto assignment.

### `BpodSystem.Data.ITI`

The ITI used for each completed trial.

### `BpodSystem.Data.TrialSettings`

The synced GUI settings used to build each completed trial.

### Raw Events

`AddTrialEvents` stores raw Bpod states and events for each completed trial. The plotter uses those raw events to draw state and event timing.
