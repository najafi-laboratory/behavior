# Function Reference

This page explains what each MATLAB source file and helper function does in the protocol. The descriptions use simple workflow language so the protocol can be debugged or modified without first reading every line of code.

## `block_single_interval_discrimination_202606.m`

### `block_single_interval_discrimination_202606`

Main protocol entry point. It removes protocol subfolders from the MATLAB path, loads GUI defaults, initializes Bpod session data, hardware, stimulus display, and servo objects, then generates the planned trial arrays. The trial loop runs one trial at a time:

1. Sync the GUI parameters.
2. Recompute the current trial's opto column from the current GUI settings.
3. Sample ISI, ITI, and punish ITI for this trial.
4. Build and load the visual/audio stimulus.
5. Build, send, and run the Bpod state machine.
6. Decode outcome and events from `RawEvents`.
7. Save trial fields into `BpodSystem.Data`.
8. Update the live plots.

At the end, cleanup returns the display to grey, moves the spouts out, closes figures, and releases hardware objects.

### `initializeSessionData`

Creates `BpodSystem.Data` fields that this protocol expects. It also keeps a safe `RigName` value so standard Bpod save code does not fail when rig metadata is missing.

### `initializeHiFi(S)`

Opens the HiFi module if a port is available. It first releases stale serial objects on the same port, then creates `BpodHiFi`, sets sampling rate, and applies attenuation from the GUI. If no valid HiFi port is present, audio-only or audio+visual modes cannot deliver sound correctly.

### `releaseSerialPort(portName)`

Searches MATLAB serialport objects for a matching port and deletes them. This prevents old, half-open handles from blocking a new connection.

### `initializeVideo(~)`

Creates the Psychtoolbox video player, configures the sync patch and timer mode, and loads the ready grey video. The grey movie is the same visual state used during ISI and before the second Enter prompt.

### `initializeServo(S)`

Finds and opens the Pololu Maestro servo controller, stores it in `BpodSystem.PluginObjects.Servo`, and moves the spouts to their out position before trials begin.

### `loadTrialStimulus(stimulus)`

Loads the current trial's video frames into the video player and loads the current audio waveform into HiFi. It is called after ISI is sampled, because stimulus duration and audio timing depend on that trial's interval.

### `showGrayScreen`

Sends the soft code that plays the grey ready movie. This keeps the stimulus display in the same grey state used between trials.

### `waitForGrayScreenEnter`

Prepares the grey ready screen and waits for the second Enter key before the first trial. During the wait, it repeatedly refreshes or replays the grey screen so clicking another window does not leave the stimulus monitor on the startup display.

### `prepareReadyGrayScreen`

Loads the ready grey movie and starts it before the second Enter prompt. It is separated from `waitForGrayScreenEnter` so the grey-state setup can be retried cleanly.

### `holdReadyGrayScreen`

Keeps the grey ready screen active while MATLAB waits for user confirmation. It checks the stimulus window repeatedly and reissues the grey display when needed.

### `forceGrayScreen`

Forces the stimulus display back to grey using the same pathway as the runtime ISI grey display. This is used before session start and during cleanup.

### `loadReadyGrayVideo`

Builds or reloads the grey video asset used for the ready screen. The goal is to have a known neutral frame available even before trial-specific stimulus videos are loaded.

### `removeProtocolSubfoldersFromPath`

Removes local reference/document folders from MATLAB's path. The protocol should run from its own files only, not from `ref1`, `ref2`, or documentation folders.

### `positionParameterGUI`

Moves the Bpod parameter GUI to a consistent screen position. This is a usability helper and does not affect trial logic.

### `validateSettings(S)`

Checks GUI values before trials run. It catches impossible settings such as invalid fractions, negative durations, bad block lengths, invalid ISI ranges, and other values that would make trial generation or the state machine unsafe.

### `printTrialInfo(...)`

Prints a compact trial summary to the MATLAB command window. It includes trial type, block type, probe type, opto periods, target side, sampled ISI/ITI values, and stimulus source.

### `sourceName(useSavedImage)`

Returns a short label for the visual source: saved image or generated grating.

### `optoPeriodText(optoType)`

Converts the current trial's opto column into readable text. The three rows are stimulus, choice, and reward; enabled rows are joined for the trial log.

### `trialTarget(S, trialType)`

Maps short/long trial type to the correct side according to `Contingency`. It also chooses the correct lick event, wrong lick event, valve output, and reward amount. If contingency is short-left/long-right, short trials target left and long trials target right; the reverse contingency swaps those assignments.

### `sampleTrialISI(S, trialType)`

Samples the current trial's ISI using the short or long ISI GUI section. Fixed mode returns the fixed value. Uniform mode returns:

```text
min + rand * (max - min)
```

### `sampleTrialITI(S)`

Samples the normal ITI using `sampleITIValue` and the ITI GUI values.

### `sampleTrialPunishITI(S)`

Samples the punish ITI using `sampleITIValue` and the punish ITI GUI values.

### `sampleITIValue(mode, manualValue, minimum, maximum, meanValue)`

Returns either a manual ITI or a bounded exponential random value. The exponential draw is truncated to `[minimum, maximum]`. In simple terms, shorter values are more common, but no value can be below the minimum or above the maximum.

### `sampleValue(mode, fixedValue, minimum, maximum)`

Generic fixed/uniform sampler used for simple interval parameters. Fixed mode returns `fixedValue`; uniform mode draws evenly between `minimum` and `maximum`.

### `trialOutcome(rawEvents, target, probeType)`

Computes the trial outcome from Bpod events. Probe trials are not scored as normal choices. For trained trials, a correct lick gives reward, a wrong lick gives wrong-side unless the change-of-mind rescue state was visited, and no lick during the choice window gives no-choice.

### `rawStates(rawEvents)`

Safely extracts the `States` structure from one trial of Bpod `RawEvents`. If the field is missing, it returns an empty structure so plotting code can fail gently.

### `rawEventData(rawEvents)`

Safely extracts the `Events` structure from one trial of Bpod `RawEvents`.

### `stateVisited(states, name)`

Returns true when a state has at least one finite timestamp. Bpod stores unvisited states as `NaN`, so the function checks for real numeric times.

### `eventInState(rawEvents, eventName, stateName)`

Checks whether an event occurred inside a named state's time interval. It is used to classify licks relative to the choice or rescue windows.

### `moveSpoutsOut(S)`

Commands the servo to the out position. It is used during startup and cleanup so the animal is not left with the spouts in.

### `findMaestroPort`

Finds the serial port for the Pololu Maestro controller, using Windows device information when possible.

### `findRegistryPorts(deviceKey)`

Queries Windows registry-style device information for candidate COM ports. This supports automatic servo detection.

### `position = maestroPosition(value)`

Converts the GUI servo position value into the Maestro command units used by the controller.

### `printSessionSummary(completedTrials, S)`

Prints a short end-of-session summary, including completed trial count and session settings that matter for interpreting the run.

### `cleanupProtocol`

Runs the end-of-session reset. It stops opto timers, returns the screen to grey, moves spouts out, closes session windows, and releases hardware objects where possible.

### `closeSessionFigures`

Closes protocol-owned figures such as the plot canvas, outcome legend, and stimulus display windows. It avoids closing unrelated MATLAB figures.

## `ConfigureProtocol.m`

### `ConfigureProtocol(BpodSystem)`

Builds the GUI parameter structure. It defines default values, popup menus, checkboxes, and panel organization for Session, Stimulus, ISI, Manipulation, Probe, Choice, Reward, Servo, and ITI settings. The GUI values are later synced at the start of every trial so user changes can affect future trials.

## `GenerateTrials.m`

### `GenerateTrials(S)`

Creates the planned session arrays: `trialTypes`, `blockTypes`, `blockStarts`, and `blockEnds`. Trial type `1` is short and trial type `2` is long. ISI, ITI, and punish ITI are sampled trial by trial in the main protocol so GUI changes can affect later trials.

### `generateBlocks(S, nTrials)`

Builds the block layout. Each block has a start trial, end trial, and block type. Block lengths are based on `BlockLength` plus or minus a random margin. Warmup blocks are always 50/50.

### `blockTypeForIndex(S, blockIndex, previousType)`

Chooses the block type after warmup. With one block mode, all blocks are 50/50. With two block mode, the protocol starts with required and extra 50/50 warmup blocks, then alternates short-majority and long-majority blocks. With three block mode, it starts with the same warmup blocks, then uses 50/50, short-majority, and long-majority blocks without repeating the previous type.

### `leadingFiftyFiftyBlocks(S)`

Returns the number of leading 50/50 blocks in block modes 2 and 3. The count is one required 50/50 block plus `WarmupBlockNum` additional 50/50 blocks.

### `sampleBlockTrials(S, blockType, nTrials)`

Samples short/long trial identities inside one block. In 50/50 blocks, short and long are sampled uniformly. In majority blocks, the majority trial type is chosen with probability `MostFraction`. The first and last `BlockEdgeTrials` of a majority block are forced to the majority type to make block edges clear.

## `GenerateProbeTrials.m`

### `GenerateProbeTrials(S, blockTypes)`

Creates a probe type vector with one value per trial. Probe type `0` is normal, `1` is stimulus-only, and `2` is servo-only. Probe trials are sampled only when probe mode is enabled.

### `eligibleTrials(S, blockTypes)`

Returns trials that may become probes. It excludes the first and last `ProbeZeroEdgeTrials` in each block so probes do not land at block edges.

## `GenerateOptoTrials.m`

### `GenerateOptoTrials(S, blockTypes, blockStarts, blockEnds)`

Creates the initial intended opto schedule as a `3 x nTrials` matrix. Rows are stimulus, choice, and reward periods. Columns are trials. A column of all zeros means opto off. A column may contain more than one `1`, which means that trial will use an arbitrary combination of enabled periods.

The schedule is used to show small intended markers in the opto plot before trials are completed. The actual current-trial column is regenerated at trial start so mid-session GUI changes are respected.

## `GenerateOptoTrial.m`

### `GenerateOptoTrial(S, blockTypes, blockStarts, blockEnds, trial)`

Generates the current trial's opto column from the current GUI settings. Warmup 50/50 blocks are never opto. If the current trial is selected for opto, the function copies the enabled period checkboxes into the three-row output column.

### `isRandomOptoTrial(S, blockStart, blockEnd, trial)`

Implements random opto mode. It excludes the first and last `OptoZeroEdgeTrials` in a block and then tags eligible trials with probability `OptoFraction`.

### `leadingFiftyFiftyBlocks(S)`

Returns the block count excluded from opto scheduling: the required first 50/50 block plus any additional `WarmupBlockNum` blocks.

### `earlyTrialsInBlock(S, blockStart, blockEnd)`

Returns the first `OptoEarlyTrials` trials of a block. Early-trial opto modes use this list.

### `blockEdges(blockTypes)`

Computes block starts and ends from a block-type vector when explicit block edge arrays are not supplied.

## `OptoControl.m`

### `OptoControl(action, S, varargin)`

Central opto helper. With action `build`, it returns global timer setup and output actions for the current trial. With display-related actions, it returns period labels and timer IDs used by plots.

### `buildActions(S, optoType, stimulusPeriod_s)`

Builds Bpod global timer actions for selected opto periods. The three periods are:

- Stimulus: from `AudStimTrigger` onset through spout-in offset.
- Choice: from `ChoiceWindow` onset to choice-window offset.
- Reward: during `PostRewardDelay`.

Each selected period uses `PWM1` as the output. The timers can span multiple states, so the state machine starts or cancels timers explicitly instead of assuming opto is contained within one state.

### `offSpecs(timerIDs)`

Creates timer definitions that keep all opto timers safely off when a trial has no enabled opto periods.

### `gateSpec(timerID, duration)`

Creates one timer definition for one opto gate. The timer drives `PWM1` high for `duration` seconds, then turns it off.

### `optoTimerIDs`

Returns the fixed global timer IDs used for stimulus, choice, and reward opto periods.

### `timerCancelMask(timerIDs)`

Computes the bitmask used by Bpod to cancel several global timers with one `GlobalTimerCancel` action. This avoids duplicate cancel actions in one state.

## `BuildStimulus.m`

### `BuildStimulus(S, isi)`

Builds the current trial's stimulus package. It creates visual frames, sync-patch frames, and an audio waveform with the same total duration. Stimulus mode controls whether visual frames, audio, or both are used.

### `loadBaseFrames(gratingFrame, grayFrame)`

Creates base video frames from the stimulus image and the neutral grey frame.

### `stimulusImage(width, height, S)`

Chooses the visual image source. If `UseSavedImage` is checked, it reads `image.png`; otherwise it generates a grating.

### `savedImage(width, height)`

Reads `image.png`, converts it to the display size, and returns the frame used for visual stimulation.

### `gratingImage(width, height)`

Generates a simple grating frame. This is the fallback visual stimulus when no saved image is used.

### `audioStimulus(S, gratingDuration, isi, sampleRate)`

Builds the audio waveform. It places tone pulses around the ISI so audio and visual timing match.

### `sineTone(frequency, duration, volume, sampleRate, rampMs)`

Generates one sine tone with onset and offset ramps. The ramp avoids sharp clicks by smoothly increasing and decreasing amplitude.

## `BuildStateMachine.m`

### `BuildStateMachine(S, stimulusDuration, iti, punishITI, target, probeType, optoType)`

Creates the Bpod state machine for one trial. It defines stimulus playback, grey screen return, spout movement, choice handling, reward, punishment, ITI, and opto timers.

### `nextAfterStimulus(probeType)`

Chooses the state after stimulus. Stimulus-only probes go directly to ITI; normal and servo-only trials continue toward spout movement.

### `nextAfterSpoutIn(S)`

Chooses the state after the spout-in command. Naive mode proceeds to auto-reward logic; trained mode proceeds to the choice window.

### `addNaiveStates(sma, S, target, opto)`

Adds the naive workflow: spout in, water delivery, wait for correct lick, post-reward delay, spout out, ITI. Naive mode is designed for shaping, so it does not punish wrong choices the same way trained mode does.

### `addTrainedStates(sma, S, target, opto)`

Adds the trained workflow: spout in, choice window, reward path for correct choices, wrong-side path for incorrect choices, optional change-of-mind rescue, punish ITI, and final ITI.

### `stimulusOptoDuration(S, stimulusDuration, probeType)`

Computes the stimulus-period opto duration. This period starts at `AudStimTrigger` and extends until spout-in offset for normal trials. For stimulus-only probe trials, it is limited to the stimulus path because the spout never moves in.

## `SoftCodeHandler_BlockSingleInterval.m`

### `SoftCodeHandler_BlockSingleInterval(code)`

Receives soft codes from the Bpod state machine. Codes control video playback and servo movement.

### `moveSpoutsIn`

Moves the left and right spouts to their in positions with the Maestro controller.

### `moveSpoutsOut`

Moves both spouts to their out positions.

### `maestroPosition(value)`

Converts a GUI servo value into Maestro command units.

### `playVideo(index)`

Plays a loaded video by index on the stimulus display.

### `playGray`

Plays the grey ready/ISI video.

### `stopVideo`

Stops stimulus video playback.

## `ProtocolPlot.m`

### `ProtocolPlot(action, trialTypes, blockTypes, probeTypes, optoTypes, isiValues, completedCount, S)`

Creates or updates the live plot canvas. On `init`, it creates the session figure and axes. On update, it redraws the schedule plots, outcome summaries, lick traces, reaction time, state timing, and event plot.

### `initializeAxes`

Applies shared axis style to valid axes only. This avoids errors when an axis has already been deleted by the user.

### `closeOutcomeLegendFigure`

Closes the separate outcome legend figure before creating a new one.

### `updatePlots(...)`

Main plot dispatcher. It calls all individual plot update functions in a consistent order.

### `updateTrialTypes`, `updateBlockTypes`, `updateProbeTypes`

Draw the short/long trial schedule, block schedule, and probe schedule. Completed trials are shown with stronger markers; future planned values are shown more lightly.

### `updateOptoTypes`

Draws opto off/stimulus/choice/reward rows. Future intended settings appear as small dots, and each trial is replaced with the actual trial-start setting as the session progresses.

### `assignedOptoCount`

Determines how many opto columns have been assigned during the running session. It uses `BpodSystem.Data.AssignedOptoTrialCount` when available.

### `plotOptoMarkers`

Draws opto markers for one group of trials. Off trials and enabled period rows are all represented so the schedule is visible even when opto is disabled.

### `optoRows`

Maps a three-row opto column into y-axis rows: off, stimulus, choice, and reward.

### `drawTrialTypeOutcome`

Overlays trial outcomes onto the trial-type plot. Reward, wrong-side, no-choice, and change-of-mind outcomes use the shared outcome color set.

### `drawOutcomeTrials`

Draws one outcome class onto the trial-type axis.

### `updateISI`

Plots ISI values as they are sampled trial by trial. Future trials are blank until the runtime sampler assigns a value.

### `updateOutcomeSummary`

Draws the session-level outcome percentages separately for short and long trials. Percentage is:

```text
100 * outcome_count / completed_trials_of_that_trial_type
```

### `updateBlockOutcomeSummary`

Draws outcome percentages by block type and trial type. Rows are 50/50, short-majority, and long-majority blocks; each row separates short and long trials.

### `drawOutcomeLegend`

Draws the independent outcome legend shared by the outcome summary plots.

### `updateLickTraces`

Updates lick density plots for short and long trials.

### `collectChoiceLicks`

Collects left and right lick times relative to choice-window onset. Times are binned on the plot x-axis.

### `drawLickDensity`

Plots left and right lick density. The rate is smoothed counts divided by trial count and bin width:

```text
lick_rate = smoothed_count / n_trials / bin_width
```

### `updateReactionTime`

Updates the reaction-time scatter and box plot. Reaction time is:

```text
first lick after choice-window onset - choice-window onset
```

### `reactionTimesByType`

Collects reaction times separately for short and long trials.

### `firstChoiceLickTime`

Finds the first valid left or right lick after choice-window onset.

### `drawReactionBox`

Draws a simple box summary for reaction times: median, quartiles, and whisker range.

### `percentileValue`

Computes a percentile by sorting values and interpolating between neighbors.

### `smoothCounts`

Smooths histogram counts with a short symmetric kernel. This keeps lick-density lines readable without a complex model.

### `outcomePercentages`

Computes outcome percentages by trial type for the session-level outcome plot.

### `blockOutcomePercentages`

Computes outcome percentages by block type and trial type for the block outcome plot.

### `outcomeColors`, `choiceColors`, `trialTypeColors`, `neutralColor`, `neutralScheduleColors`, `futureColor`

Return the color palettes used consistently across plots. Outcomes, left/right choice, and short/long trial type each have their own color set.

### `drawSchedule`

Shared schedule plotting helper. It draws completed and future trial values with y labels, x ticks, and a visible trial window.

### `updateDetailPlots`

Updates state timing and event plots for the latest completed trial.

### `updateStateTiming`

Draws the timing of visited Bpod states for one trial.

### `showEvents`

Draws licks, BNC events, and LED/opto intervals for one trial.

### `optoIntervals`

Reconstructs LED1/opto-on intervals from saved opto period settings and state timing. This lets the event plot show opto light even when the hardware event itself is not recorded as a standard input event.

### `trialOptoType`

Reads the saved opto column for one trial from Bpod data.

### `appendInterval`

Adds a valid interval to an interval list after clipping it to the plotted trial duration.

### `stateStart`, `stateEnd`, `firstStateInterval`, `firstFinite`

Small helpers for extracting valid state times from Bpod state arrays.

### `eventIntervals`, `eventTimes`

Extract event times or convert on/off event pairs into plotted intervals.

### `drawIntervals`

Draws horizontal event intervals on the event axis.

### `showEmpty`

Displays a readable placeholder when no data are available yet.

### `visibleWindow`, `trialTicks`

Choose the x-axis trial range and tick marks for schedule plots.

### `trialDuration`

Computes a safe plotted duration for a trial from its state timestamps.

## `PololuMaestro.m`

### `PololuMaestro`

Small class wrapper for the Pololu Maestro serial protocol.

### `PololuMaestro(portString)`

Constructor. Opens the Maestro serial connection through `ArCOMObject_Bpod`, then initializes channels 0 through 5 with default speed and acceleration values.

### `setMotor(obj, motorID, position, velocity, acceleration)`

Moves one Maestro channel. `position` is expected on the protocol's normalized servo scale. `velocity` and `acceleration` are clipped to `[0, 1]`, converted to Maestro speed/acceleration values, and only resent when they changed. The target is converted to Maestro quarter-microsecond units before sending.

### `delete(obj)`

Releases the serial object by clearing `obj.Port`.

### `setSpeed(obj, channel, value)`

Private helper that sends the Maestro speed command for one channel.

### `setAcceleration(obj, channel, value)`

Private helper that sends the Maestro acceleration command for one channel.

### `writeCommand(obj, command, channel, value)`

Private low-level serial writer. Maestro commands split a numeric value into low and high 7-bit bytes:

```text
lowBits = value & 127
highBits = value >> 7
```

The command, channel, and two value bytes are then written to the serial port as `uint8`.
