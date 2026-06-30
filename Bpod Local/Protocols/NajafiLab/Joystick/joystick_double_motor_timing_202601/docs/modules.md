# Modules And Files

This page explains each active file.

## `joystick_double_motor_timing_202601.m`

This is the main protocol file.

It does the session workflow:

1. Load settings.
2. Open GUI.
3. Validate settings.
4. Confirm opto settings.
5. Open hardware.
6. Build visual cues.
7. Wait for the start key.
8. Generate trial schedules.
9. Run each Bpod trial.
10. Save data.
11. Update plots.
12. Clean up hardware.

It also contains helper functions for:

- Rig detection.
- Serial port cleanup.
- Rotary encoder connection.
- Maestro port detection.
- GUI positioning.
- Probe trial settings.
- Setting validation.
- Trial result printing.
- Session summary printing.
- Outcome classification.

## `ConfigureProtocol.m`

This builds the protocol settings.

It defines:

- GUI default values.
- GUI panel groups: `Session`, `Stimulus`, `Timing`, `Joystick`, `Reward`, `ITI`, and `Manipulation`.
- Popup menu labels.
- Checkbox fields.
- Config migration rules.

It also removes old settings that are no longer used.

## `GenerateTrials.m`

This makes three vectors:

- `trialTypes`
- `itiValues`
- `punishITIValues`

`trialTypes` contains:

- `1`: short trial.
- `2`: long trial.

ITI values can be manual or exponential.

Block modes alternate between short and long blocks.

## `GenerateVisualCueVideo.m`

This builds the cue frame.

It either:

- Loads `image.png`.
- Or creates a generated grating.

The frame is resized to the display viewport.

The cue duration is rounded to a whole number of display frames.

## `BuildStateMachine.m`

This builds the Bpod state matrix for one trial.

Inputs include:

- Current settings.
- Short or long delay.
- Trial type.
- Opto type.
- Assist status.
- ITI.
- Punish ITI.

The state matrix handles:

- Cue 1.
- Servo release.
- Press 1.
- Press 2.
- Pre reward delay.
- Reward.
- Post reward delay.
- Error states.
- ITI.

It also adds opto output actions for the current opto period column.

## `SoftCodeHandler_Protocol.m`

This handles soft codes from the state machine.

It controls:

- Visual cues.
- Gray screen.
- Servo movement.
- Press 2 timing.
- Reward calculation.
- Valve opening.
- Outcome soft-code return.

Dynamic reward is computed here.

## `OptoControl.m`

This controls opto trial tags and opto actions.

Opto is saved as a `4 x nTrials` matrix. Each column is one trial. The rows are:

- Row 1: cue 1 period.
- Row 2: delay / press 2 period.
- Row 3: pre reward delay period.
- Row 4: post reward period.

A column of `[0; 0; 0; 0]` means opto is off for that trial. A column such as `[1; 1; 0; 0]` means the same trial gets light in cue 1 and delay, but not pre reward delay or post reward.

It also returns plot labels and neutral grayscale colors.

When `OptoMode` is off, every trial receives `[0; 0; 0; 0]`.

When `OptoMode` is on, `OptoControl('trials', S, trialTypes, probeTypes)` first makes an intended schedule for the whole session. It excludes the first block, excludes the first and last `OptoZeroEdgeTrials` trials of each later block, and excludes every probe trial. It then tags approximately `OptoFraction` of the eligible non-probe trials.

Only checked opto periods can be assigned:

- `EnableOptoVisualCue1` sets row 1.
- `EnableOptoDelay` sets row 2.
- `EnableOptoPreRewardDelay` sets row 3.
- `EnableOptoPostReward` sets row 4.

If more than one period is checked, a selected opto trial gets all checked periods. The protocol no longer chooses only one period.

At the start of each trial, after the GUI is synced, `OptoControl('trial', S, trialTypes, currentTrial, probeTypes)` assigns the current trial again using the current GUI values. Probe trials always receive an all-zero opto column. That current opto column overwrites the intended schedule for that trial. This means changes made between trials affect the next trial without changing completed trials.

The intended schedule plus online overwrites are saved in `BpodSystem.Data.PlannedOptoTrialTypes`. Completed assigned columns are saved in `BpodSystem.Data.OptoTrialTypes`. `BpodSystem.Data.AssignedOptoTrialCount` stores the highest trial index assigned online.

`OptoControl('actions', S, optoType, delay, press2Window)` returns Bpod output actions for the current trial:

- Row 1 on: LED1 turns on in `VisualStimulus1` and turns off when cue 1 exits.
- Row 2 on: global timer 10 starts in `LeverRetract1`, drives `PWM1` high through the delay / press 2 period, and is cancelled at `RewardLeverRetract`.
- Row 3 on: LED1 turns on in `PreRewardDelay` and turns off in `Reward`.
- Row 4 on: LED1 turns on in `PostRewardDelay` and turns off in `LeverRetractFinal`.

All enabled rows in a column can run in the same trial. For example, `[1; 0; 0; 1]` gives light during cue 1 and post reward. `[1; 1; 1; 1]` gives light in all four periods.

The current implementation uses continuous LED1 high intervals for each selected period. `OptoFrequency_Hz` and `OptoPulseOn_ms` are GUI parameters reserved for pulsed stimulation, but the active state-machine action sends a sustained `PWM1` high value during each selected period.

The event plot reconstructs the LED1 row from `OptoTrialTypes` and state timing. This keeps plotting self-contained and does not require reading a separate reference implementation.

## `ProbeControl.m`

This controls probe trial tags and probe metadata.

It supports:

- `0`: Off.
- `1`: Reward omitted.
- `2`: Timing mode flipped.

It also returns plot labels and neutral grayscale colors.

## `ProtocolPlot.m`

This manages the online session figure.

It updates:

- Trial type raster.
- Opto type raster.
- Probe type raster.
- Short press 2 timing.
- Long press 2 timing.
- Completed outcome fractions.
- All outcome fractions.
- Outcome legend.
- Delay values.
- Encoder trace.
- State timing.
- BNC, LED, and lick events.

It also reconstructs LED1 intervals for plotting.

Outcome-related plots share the outcome palette. Opto, probe, delay, and event plots use neutral gray or black marks. State timing and encoder annotations use colored markers, and the rotary encoder position trace is black.
