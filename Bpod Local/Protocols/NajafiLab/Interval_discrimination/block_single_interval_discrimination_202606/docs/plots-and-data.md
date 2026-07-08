# Plots And Saved Data

## Live Canvas

`ProtocolPlot.m` creates a tiled session canvas.

Plots include:

- Trial type and outcome markers.
- ISI values, updated trial by trial.
- Block type.
- Probe type.
- Opto period raster with separate off, stimulus, spout-in-delay, spout-in, choice, pre-outcome, reward, post-reward, and punish-ITI rows. Future trials show the initial intended schedule as small dots; each trial is overwritten with the actual opto settings when it is assigned after GUI sync.
- Outcome percentage by trial type.
- Outcome percentage by block and trial type.
- Lick density for short and long trials.
- Reaction time by trial type.
- State timing for the latest completed trial.
- Event raster for lick ports, BNC1, BNC2, and LED1. LED1 is reconstructed from `OptoTrialTypes` and raw state timing, with raw `PWM1High/PWM1Low` events used only as a fallback.

## Plot Color Semantics

- Left/right choice colors are used for lick traces and Port1/Port3 events.
- Short/long trial colors are used for trial-type related plots.
- Outcome colors are used for outcome markers and outcome bars.
- Neutral greys are used for hardware and non-semantic tags.

## Saved Fields

The protocol saves standard Bpod raw events plus derived fields including:

- `TrialTypes`
- `BlockTypes`
- `ProbeTrialTypes`
- `OptoTrialTypes`
- `AssignedOptoTrialCount`
- `ChemoTrialTypes`
- `ChemoTrialType`
- `ISI`
- `ITI`
- `PunishITI`
- `CorrectSide`
- `Contingency`
- `Outcomes`
- `StimulusDuration`
- `StimulusMode`
- `UseSavedImage`

Planned vectors are also saved before trials begin:

- `PlannedTrialTypes`
- `PlannedBlockTypes`
- `PlannedBlockStarts`
- `PlannedBlockEnds`
- `PlannedProbeTrialTypes`
- `PlannedOptoTrialTypes`
- `PlannedISI`
- `PlannedITI`
- `PlannedPunishITI`
- `PlannedChemoTrialTypes`
