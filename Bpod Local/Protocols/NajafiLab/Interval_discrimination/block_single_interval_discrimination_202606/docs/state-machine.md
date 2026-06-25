# State Machine

The state machine is built in `BuildStateMachine.m`.

## Shared Start

All normal trials begin:

```text
Start
VisStimTrigger
AudStimTrigger
StimulusDone
```

`StimulusDone` returns the display to grey and stops HiFi.

If stimulus opto is enabled for the trial, `AudStimTrigger` starts an opto global timer. The timer duration is computed so `PWM1` stays high until the end of spout-in movement on normal trials.

## Normal Trained Trial

```text
SpoutInDelay
SpoutIn
ChoiceWindow
```

`SpoutIn` sends the servo-in soft code at state onset. It transitions only on `Tup`, so `ChoiceWindow` begins after `ServoMoveDelay_s`, not at servo command onset.

If choice opto is enabled, `ChoiceWindow` starts a global timer. `PreRewardDelay`, `ChangeMindWindow`, and servo-out paths cancel that timer so `PWM1` follows the actual choice-window offset.

Correct lick:

```text
PreRewardDelay
Reward
PostRewardDelay
ServoOut
ITI
```

If pre-reward opto is enabled, `PreRewardDelay` starts a global timer that keeps `PWM1` high through `Reward` offset. If post-reward opto is enabled, `PostRewardDelay` starts a separate global timer that keeps `PWM1` high for `PostRewardDelay_s`.

Wrong lick without change-of-mind:

```text
ServoOutPunish
PunishITI
ITI
```

If punish-ITI opto is enabled, `PunishITI` starts a global timer that keeps `PWM1` high for the sampled punish ITI duration.

Wrong lick with change-of-mind:

```text
ChangeMindWindow
```

If the correct side is licked during `ChangeMindWindow`, the trial follows the reward path. Otherwise it follows the punish path.

## Naive Trial

```text
SpoutInDelay
SpoutIn
NaiveReward
WaitForCorrectLick
PostRewardDelay
ServoOut
ITI
```

Naive trials deliver reward after spouts move in, then wait for the correct lick.

## Probe Trials

Stimulus-only probe:

```text
StimulusDone
ITI
```

Servo-only probe:

```text
StimulusDone
ProbeSpoutIn
ProbeChoiceWindow
ServoOut
ITI
```

Probe trials can carry opto tags. Stimulus-only probes only use the stimulus duration for the stimulus opto period. Servo-only probes use the stimulus duration plus servo-in movement for the stimulus opto period.
