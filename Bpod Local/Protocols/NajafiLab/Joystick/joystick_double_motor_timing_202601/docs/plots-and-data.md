# Plots And Data

This page explains online plots and saved data.

## Online Plot Layout

The online plot is created by `ProtocolPlot.m`.

It uses one two-column figure.

The left column uses about 60% of the plotting canvas and shows, from top to bottom:

- Trial type raster.
- Opto type raster.
- Probe type raster.
- Trial delay values.
- Rotary encoder trace.

The right column shows, from top to bottom:

- Completed outcome fraction, all-outcome fraction, and the outcome legend.
- Short and long press 2 timing histograms.
- State timing.
- BNC, LED, and lick events.

Outcome-related plots share one color set for reward, early, late, no press 2, no press 1, and assist outcomes. Opto, probe, delay, and event plots use neutral gray or black marks. The encoder position trace is black, while encoder annotations and state timing use colored markers.

## Trial Type Raster

This plot shows short and long trial type.

It also marks completed outcomes.

Marker colors show:

- Reward.
- Press 2 early.
- Press 2 late.
- Did not press 2.
- Did not press 1.
- Assist trial.

## Opto Raster

This plot shows opto trial type.

Values are:

- `0`: Off.
- `1`: Visual cue 1.
- `2`: Delay.
- `3`: Post reward.

## Probe Raster

This plot shows probe trial type.

Values are:

- `0`: Off.
- `1`: Reward omitted.
- `2`: Timing mode flipped.

## Press 2 Timing Histograms

There are two timing plots.

One is for short trials.

One is for long trials.

Each plot shows press 2 time relative to perfect timing.

The x axis is:

```text
press 2 time - target delay
```

The histogram bars use the same outcome colors as the outcome rasters and fractions.

The reward window profile and perfect timing marker use neutral gray or black.

## Completed Outcome Fraction

This plot summarizes completed press 2 outcomes.

It is split by short and long trials.

It includes:

- Reward.
- Press 2 early.
- Press 2 late.
- Assist trial.

It does not include no-press outcomes.

It displays:

- Stacked fractions.
- Percent labels for large segments.
- `n=` count for each trial type.

## All Outcome Fraction

This plot summarizes all terminal outcomes.

It is split by short and long trials.

It includes:

- Reward.
- Press 2 early.
- Press 2 late.
- Did not press 2.
- Did not press 1.
- Assist trial.

It displays:

- Stacked fractions.
- Percent labels for large segments.
- `n=` count for each trial type.

## Outcome Legend

The legend lists all outcome colors.

It includes assist trials.

## Delay Plot

This plot shows the delay used on each completed trial.

Short and long trials use neutral gray and black marks.

## Encoder Plot

This plot shows the rotary encoder trace for the last completed trial.

It includes:

- Position.
- Zero line.
- Press threshold.
- Retract threshold.
- State event markers.
- Cue 2 or perfect timing marker.

The position trace is black. Thresholds and event markers use colored annotations.

## State Timing Plot

This plot shows raw Bpod states for the last completed trial.

Each state is one row.

The x axis is time in the trial.

The plot has x grid lines and y grid lines.

## Event Plot

This plot shows digital events for the last completed trial.

Rows are:

- BNC1.
- LED1.
- Port 1 lick.

LED1 is reconstructed from opto type and state timing.

## Saved Data Fields

The protocol saves standard Bpod data plus custom fields.

Important fields include:

- `TrialSettings`
- `TrialTypes`
- `TrialTransitions`
- `OptoTrialTypes`
- `ProbeTrialTypes`
- `ProbeRewardOmitted`
- `ChemoTrialTypes`
- `AssistTrial`
- `Press2Time`
- `RewardAmount`
- `ITI`
- `PunishITI`
- `EncoderData`
- `PlannedTrialTypes`
- `PlannedOptoTrialTypes`
- `PlannedProbeTrialTypes`
- `PlannedITI`
- `PlannedPunishITI`

## Encoder Data

Each trial can save one encoder data entry.

The protocol normalizes time so each trace starts at zero.

The protocol normalizes position so each trace starts at zero.

Fields include:

- `Times`
- `Positions`
- `EventTimestamps`
- `nPositions`
