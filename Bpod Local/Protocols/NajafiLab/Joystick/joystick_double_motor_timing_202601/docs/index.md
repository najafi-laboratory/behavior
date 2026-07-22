# Joystick Double Motor Timing 202601

This site documents the Bpod protocol `joystick_double_motor_timing_202601`.

The protocol trains a mouse to press a joystick at a target time. It supports short and long timing trials, visual-guided and self-timed modes, and visual-only, audio-only, or audio+visual sensory cues. It also supports optogenetic trials, probe trials, assist trials, dynamic reward, rotary encoder logging, and online plots.

## Start Here

- [Main workflow](workflow.md)
- [Parameters](parameters.md)
- [State machine](state-machine.md)
- [Modules and files](modules.md)
- [Plots and data](plots-and-data.md)
- [Operation checklist](operation-checklist.md)
- [GitHub Pages deployment](deployment.md)

## Main Idea

Each trial has a target press time.

Short trials use `ShortDelay_s`.

Long trials use `LongDelay_s`.

The mouse presses the joystick. The protocol measures the press 2 time. Reward size depends on how close press 2 is to the target time.

## Trial Outcomes

The protocol records these outcomes:

- `LeverRetractFinal`
- `EarlyPress2`
- `Press2Late`
- `DidNotPress1`
- `DidNotPress2`
- `AssistTrial`

## Hardware

The protocol uses these hardware parts:

- Bpod state machine.
- Rotary encoder module.
- Pololu Maestro servo controller.
- Optional HiFi module for auditory cues.
- PsychToolbox visual display.
- Valve 2 for water reward.
- LED1 / PWM1 for optogenetic gating.
- BNC1 for trial marking.

## Active MATLAB Files

- `joystick_double_motor_timing_202601.m`
- `ConfigureProtocol.m`
- `GenerateTrials.m`
- `GenerateSensoryCueVideo.m`
- `BuildStateMachine.m`
- `SoftCodeHandler_Protocol.m`
- `OptoControl.m`
- `ProbeControl.m`
- `ProtocolPlot.m`

All active protocol code is stored in the protocol root. The protocol does not add or require a separate helper-code folder.
