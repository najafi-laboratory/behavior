# State Machine

The state machine is built in `BuildStateMachine.m`.

## Shared Start

All normal trials begin:

```text
Start
PreStimDelay
VisStimTrigger
AudStimTrigger
StimulusDone
```

`StimulusDone` returns the display to grey and stops HiFi.

`PreStimDelay` waits for `PreStimDelay_s` before visual/audio stimulus onset. If stimulus opto is enabled for the trial, this state starts the stimulus opto global timer. The timer duration is `PreStimDelay_s + stimulusDuration`, so `PWM1` stays high from pre-stimulus-delay onset until stimulus-play offset.

## Normal Trained Trial

```text
SpoutInDelay
SpoutIn
ChoiceWindow
```

`SpoutIn` sends the servo-in soft code at state onset. It transitions only on `Tup`, so `ChoiceWindow` begins after `ServoMoveDelay_s`, not at servo command onset.

If spout-in-delay opto is enabled, `SpoutInDelay` starts a global timer for `SpoutInDelay_s`. If spout-in opto is enabled, `ChoiceWindow`, `ProbeChoiceWindow`, or naive `WaitForCorrectLick` starts a global timer for the spouts-in lick window. `PreOutcomeDelay`, `ChangeMindWindow`, and servo-out paths cancel that timer so `PWM1` follows the actual spouts-in window offset.

Correct lick:

```text
PostLickDelayReward
PreOutcomeDelay
Reward
PostRewardDelay
ServoOut
ITI
```

If pre-outcome opto is enabled, `PreOutcomeDelay` starts a global timer that keeps `PWM1` high for `PreOutcomeDelay_s`. If reward opto is enabled, `Reward` starts a separate global timer for the valve-open reward duration. If post-reward opto is enabled, `PostRewardDelay` starts another timer for `PostRewardDelay_s`.

Wrong lick without change-of-mind:

```text
PostLickDelayPunish
PreOutcomeDelayPunish
ServoOutPunish
PunishITI
ITI
```

Punish trials enter `PreOutcomeDelayPunish` before `ServoOutPunish`. If pre-outcome opto is enabled, this delay starts the same pre-outcome opto timer used by reward trials.

If punish-ITI opto is enabled, `PunishITI` starts a global timer that keeps `PWM1` high for the sampled punish ITI duration.

Wrong lick with change-of-mind:

```text
PostLickDelayChangeMind
ChangeMindWindow
```

If the correct side is licked during `ChangeMindWindow`, the trial enters `PostLickDelayReward` and then follows the reward path. Otherwise it follows the punish path without a post-lick delay.

## Naive Trial

```text
SpoutInDelay
SpoutIn
NaiveReward
WaitForCorrectLick
PostLickDelayPostReward
PostRewardDelay
ServoOut
ITI
```

Naive trials deliver reward after spouts move in, then wait for the correct lick.

All `PostLickDelay...` states use `PostLickDelay_s`. If choice opto is enabled, these states start the post-lick choice opto timer.

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

Probe trials can carry opto tags. Stimulus opto uses the same pre-stimulus-delay plus stimulus-playback duration for normal, stimulus-only probe, and servo-only probe trials.
