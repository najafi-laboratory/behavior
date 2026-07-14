# Opto Test Doric 202607

This site documents the Bpod protocol `opto_test_doric_202607`.

The protocol is a minimum Doric opto timing test. It has no responses, rewards, probes, joystick, encoder, or servo dependencies. Each trial shows one image, a gray interval, then the same image again.

## Start Here

- [Main workflow](workflow.md)
- [Parameters](parameters.md)
- [State machine](state-machine.md)
- [Modules and files](modules.md)
- [Plots and data](plots-and-data.md)
- [Operation checklist](operation-checklist.md)
- [Opto setup and trigger timing](opto-setup.md)

## Trial Timeline

```text
PreStimDelay -> Image1Display -> ImageInterval -> Image2Display -> PostStimDelay -> ITI
```

Opto can occur during:

- pre-stim delay
- image interval
- image 2
- post-stim delay

## Hardware

- Bpod state machine
- PsychToolbox visual display
- `PWM1` / LED1 output for Doric gating
- `BNC1` trial marker

## Active MATLAB Files

- `opto_test_doric_202607.m`
- `ConfigureProtocol.m`
- `GenerateTrials.m`
- `GenerateVisualCueVideo.m`
- `BuildStateMachine.m`
- `SoftCodeHandler_Protocol.m`
- `OptoControl.m`
- `ProtocolPlot.m`
