# Module Responsibilities

## `block_single_interval_discrimination_202606.m`

Main protocol entry point. It configures GUI settings, initializes hardware, runs the trial loop, saves trial data, updates plots, and cleans up hardware/session windows.

## `ConfigureProtocol.m`

Defines GUI defaults, popup menus, checkboxes, and panel layout.

## `GenerateTrials.m`

Builds block types and trial types. It handles the 50/50 block, short-majority blocks, long-majority blocks, block lengths, and edge trials.

## `GenerateProbeTrials.m`

Generates probe trial tags.

## `GenerateOptoTrials.m`

Creates the initial intended opto schedule as an `8 x nTrials` matrix for stimulus, spout-in delay, spout-in, choice, pre-outcome, reward, post-reward, and punish-ITI periods.

## `GenerateOptoTrial.m`

Generates the current trial's opto column from the current GUI settings and overwrites the intended schedule for that trial. This allows opto mode, opto fraction, and enabled periods to change during a session.

## `OptoControl.m`

Defines opto global timers, output actions, and display metadata. The hardware output is `PWM1`; selected periods drive it high during their configured windows.

## `BuildStimulus.m`

Builds visual frames, sync frames, and HiFi audio. It supports generated grating or `image.png`.

## `BuildStateMachine.m`

Builds the Bpod state matrix for naive, trained, and probe trials.

## `SoftCodeHandler_BlockSingleInterval.m`

Handles servo and video soft codes.

## `ProtocolPlot.m`

Creates and updates all online plots.

## `PololuMaestro.m`

Small wrapper around the Pololu Maestro serial interface.
