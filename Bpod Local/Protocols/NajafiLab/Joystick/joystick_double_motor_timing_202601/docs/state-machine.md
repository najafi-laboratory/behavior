# State Machine

This page explains the one-trial Bpod state machine built by `BuildStateMachine.m`.

## Global Timers

### Timer 1

Timer 1 controls cue 2 timing. It starts at the press 2 timing origin, which is `PrePress2Delay`.

Its onset delay is the target delay. In visual-guided mode, it sends soft code 2 when cue 2 should appear. In self-timed mode, it sends no cue message.

### Timer 10

Timer 10 is reserved for delay-period opto. It starts in `LeverRetract1`, drives `PWM1` high during the delay / press 2 epoch, and is cancelled at `RewardLeverRetract`.

## Trial Start

`Start` turns BNC1 high, resets the rotary encoder, cancels any old opto timer, turns LED1 off, and moves to `VisualStimulus1`.

`VisualStimulus1` plays cue 1, enables encoder threshold events, and turns LED1 on when the cue 1 opto row is enabled for this trial. Double press mode goes to `WaitForPress1`; single press mode goes directly to `PrePress2Delay`.

## Press 1

`WaitForPress1` releases the servo for press 1, turns off opto type 1 when needed, and waits for the first joystick press.

If the encoder threshold is crossed, the trial goes to `Press1`.

If time runs out, the trial goes to `DidNotPress1`.

`Press1` waits for `ServoMoveDelay_s`, then goes to `LeverRetract1`.

`LeverRetract1` retracts the servo. When the soft-code handler confirms the servo is home, the trial moves to `PrePress2Delay`. Delay-period opto starts LED1 here.

## Press 2 Preparation

`PrePress2Delay` is the timing origin for press 2.

On normal trials it has zero duration, releases the servo for press 2, enables encoder events, turns off opto type 1 when needed, and moves to `WaitForPress2`.

On assist trials it holds the lever available for the target delay. The mouse can still press during this period. If no press happens by the target delay, the trial goes to `Assist`.

`Assist` releases the servo and moves to `WaitForPress2`.

`WaitForPress2` waits for the press 2 joystick event. On normal trials it uses the full press 2 window. On assist trials it uses the remaining window:

```text
press2Window - delay
```

If the encoder threshold is crossed, the trial goes to `Press2`. If time runs out, it goes to `DidNotPress2`.

## Press 2 Outcome

`Press2` sends soft code 21. The soft-code handler measures press 2 time, stops cue output, and computes reward amount. The state also cancels cue 2 timer 1.

`RewardLeverRetract` retracts the servo after press 2 and sends soft code 22. The soft-code handler returns one of three soft codes:

- `SoftCode1`: early, route to `EarlyPress2`.
- `SoftCode2`: rewarded, route to `PreRewardDelay`.
- `SoftCode3`: late, route to `Press2Late`.

If no return code arrives, the trial goes to `DidNotPress2`.

Delay-period opto is turned off in `RewardLeverRetract`.

## Reward Path

`PreRewardDelay` waits for `RewardDelay_s`, then goes to `Reward`.

`Reward` sends soft code 20. The soft-code handler delivers water through valve 2. When reward delivery is done, the handler sends soft code 3 and the trial goes to `PostRewardDelay`.

`PostRewardDelay` waits for `PostRewardDelay_s`. Post-reward opto turns LED1 on here.

`LeverRetractFinal` is the terminal rewarded state. Post-reward opto turns LED1 off here, then the trial goes to `ITI`.

## Error Path

The error outcome states are:

- `EarlyPress2`
- `Press2Late`
- `DidNotPress1`
- `DidNotPress2`

`DidNotPress1` and `DidNotPress2` also retract the servo before the punish period. All error states go to `Punish_ITI`.

`Punish_ITI` waits for the punish ITI, then goes to `ITI`.

## ITI

`ITI` waits for the normal ITI, turns BNC1 low, and exits the state machine.
