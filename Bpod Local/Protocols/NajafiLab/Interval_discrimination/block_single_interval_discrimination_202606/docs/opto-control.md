# Opto Control

## Overview

Opto control has two independent parts:

- Trial selection: decide whether the current trial is an opto trial.
- Period selection: decide which task epochs receive light on selected opto trials.

The protocol saves opto as a `5 x nTrials` matrix named `OptoTrialTypes`. Rows are stimulus, choice, pre-reward, post-reward, and punish ITI. Columns are trials. A column of all zeros means opto off. A column can contain multiple ones, meaning several periods were enabled for that trial.

Opto hardware output uses `PWM1`, labeled as `LED 1` in the event plot. The protocol gates `PWM1` high with Bpod global timers. It does not generate a pulse train; if pulsed light is needed, use downstream hardware to convert the gate.

## GUI Parameters

### `OptoMode`

This selects which trials become opto trials.

`No opto` means all trials are opto off, even if period checkboxes are checked.

`Random` samples every eligible trial online when the trial starts. Warmup blocks are excluded. The first and last `OptoZeroEdgeTrials` trials of every non-warmup block are also excluded. Eligible trials are selected with probability `OptoFraction`.

`Early trials in every block` selects the first `OptoEarlyTrials` trials of every non-warmup block.

`Early trials in alternating block groups` alternates groups of blocks after warmup. The first group is no-opto. The next group has early opto trials. The group size is `BlockNum`.

### `OptoFraction`

Used only in `Random` mode. It is a probability per eligible trial, not a fixed final session fraction. For example, `0.35` means each eligible trial has a 35% chance of being tagged opto when it starts.

Changing this value during the session changes the probability for later trials only.

### `OptoZeroEdgeTrials`

Used only in `Random` mode. It excludes block-edge trials from opto assignment. If set to `5`, the first five and last five trials of each non-warmup block are forced opto off. This is useful when block transitions should remain clean.

Warmup blocks are always opto off, independent of `OptoZeroEdgeTrials`.

### `OptoEarlyTrials`

Used in the two early-trial modes. It controls how many trials at the start of each eligible block are opto trials. If set to `5`, trials 1 to 5 within an eligible block are opto. The value is clipped by the block length.

### `EnableOptoStimulus`

Enables light during the stimulus-to-spout-in epoch on selected opto trials. The timer starts at `AudStimTrigger` onset. On normal trials it lasts until the end of `SpoutIn`, so it spans stimulus, grey screen, spout-in delay, and servo movement. On stimulus-only probe trials it lasts for the stimulus. On servo-only probe trials it lasts through `ProbeSpoutIn`.

### `EnableOptoChoice`

Enables light during the actual `ChoiceWindow` on selected opto trials. The timer starts at `ChoiceWindow` onset and is cancelled when the state exits. If the animal makes a choice early, light turns off early. If the animal does not choose, light lasts for `ChoiceWindow_s`.

### `EnableOptoPreReward`

Enables light during the reward path on selected opto trials. The timer starts at `PreRewardDelay` onset and lasts through `Reward` offset, including both `PreRewardDelay_s` and the valve-open reward duration.

### `EnableOptoPostReward`

Enables light during `PostRewardDelay` on selected opto trials. The timer starts at `PostRewardDelay` onset and lasts for `PostRewardDelay_s`.

### `EnableOptoPunishITI`

Enables light during `PunishITI` on selected opto trials. The timer starts at `PunishITI` onset and lasts until `PunishITI` offset.

## How To Use

1. Choose `OptoMode`.
2. Set the trial-selection parameters for that mode:
   - `Random`: set `OptoFraction` and `OptoZeroEdgeTrials`.
   - Early-trial modes: set `OptoEarlyTrials`.
3. Check one or more period boxes:
   - `EnableOptoStimulus`
   - `EnableOptoChoice`
   - `EnableOptoPreReward`
   - `EnableOptoPostReward`
   - `EnableOptoPunishITI`
4. Start the session.
5. Watch the opto plot:
   - small dots show the intended schedule made at session start.
   - larger solid squares show trials that have been assigned using the current GUI settings.
6. If needed, pause or change GUI settings between trials. The next trial will use the new settings.

## Online Assignment

At session start, `GenerateOptoTrials.m` creates an intended schedule for the whole session using the initial GUI settings. This is mainly for display.

At the start of each trial, after `BpodParameterGUI('sync', S)`, `GenerateOptoTrial.m` regenerates only the current trial's opto column using the current GUI values. That current column overwrites the initial schedule. Completed trials are never rewritten.

This design lets the user change opto settings during a session without rebuilding the block or trial-type plan.

## Hardware Timing

`OptoControl.m` creates five global timers:

- timer 10: stimulus period
- timer 11: choice period
- timer 12: pre-reward period
- timer 13: post-reward period
- timer 14: punish-ITI period

All timers drive `PWM1`. At trial start, the protocol cancels all opto timers and forces `PWM1` low. It also forces opto off at servo-out and ITI; `PunishITI` can then start its own selected opto timer.

Naive sessions force opto off in the state machine. Opto tags may still be saved, but no `PWM1` light is delivered in naive mode.

## Plots

The opto period plot has six rows:

- `Off`
- `Stimulus`
- `Choice`
- `PreReward`
- `PostReward`
- `PunishITI`

Small dots are the initial intended schedule. Solid squares are assigned trials. A selected trial can show more than one square if multiple periods are enabled.

The event plot has an `LED 1` row. Because Bpod global-timer PWM events are not always stored as regular `PWM1High/PWM1Low` events, the plot reconstructs LED timing from `OptoTrialTypes` and raw state timing. If raw PWM events exist, they are used as a fallback.

## Saved Data

Important saved fields:

- `PlannedOptoTrialTypes`: intended schedule plus online overwrites as trials are assigned.
- `OptoTrialTypes`: completed assigned opto columns.
- `AssignedOptoTrialCount`: highest trial index whose opto column has been assigned online.
- `TrialSettings`: full GUI settings saved per trial, useful for confirming which opto settings were active.

To inspect a trial:

```matlab
trial = 10;
BpodSystem.Data.OptoTrialTypes(:, trial)
BpodSystem.Data.TrialSettings(trial).GUI.OptoMode
BpodSystem.Data.TrialSettings(trial).GUI.EnableOptoStimulus
BpodSystem.Data.TrialSettings(trial).GUI.EnableOptoChoice
BpodSystem.Data.TrialSettings(trial).GUI.EnableOptoPreReward
BpodSystem.Data.TrialSettings(trial).GUI.EnableOptoPostReward
BpodSystem.Data.TrialSettings(trial).GUI.EnableOptoPunishITI
```

## Common Checks

If opto does not appear:

- Confirm `TrainingMode` is `Trained`; naive mode forces opto hardware off.
- Confirm `OptoMode` is not `No opto`.
- Confirm at least one period checkbox is enabled.
- In `Random` mode, confirm the trial is outside warmup blocks and outside `OptoZeroEdgeTrials`.
- Check the opto plot for assigned solid squares.
- Check the event plot `LED 1` row after the trial completes.
