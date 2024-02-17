function LeverReset
% This protocol demonstrates resetting a lever to its intial position after
% two presses during a single trial. The lever is attached to a stepper motor 
% shaft, controlled by the Bpod Stepper module. The same shaft is attached to 
% a rotary encoder and read out via the Bpod rotary encoder module.
%
% ***IMPORTANT*** The protocol must be started with the lever already in 
% its intial 'home' position.

global BpodSystem

%% Assert Stepper + Rotary Encoder modules are present + USB-paired (via USB button on console GUI)
BpodSystem.assertModule({'Stepper','RotaryEncoder'}, [1 1]); % The second argument [1 1] indicates that both HiFi and RotaryEncoder must be paired with their respective USB serial ports
BpodSystem.PluginObjects.S = BpodStepperModule(BpodSystem.ModuleUSB.Stepper1);
BpodSystem.PluginObjects.R = RotaryEncoderModule(BpodSystem.ModuleUSB.RotaryEncoder1); 

%% Define parameters
S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S
if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
    S.threshold = 2; % Threshold for completion of a lever press, units = degrees of shaft rotation
    S.GUI.currentTrial = 0; 
end

%% Initialize plots
BpodParameterGUI('init', S); % Initialize parameter GUI plugin
%-- Last Trial encoder plot (an online plot included in the protocol folder)
BpodSystem.ProtocolFigures.EncoderPlotFig = figure('Position', [500 200 350 350],'name','Encoder plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.EncoderAxes = axes('Position', [.15 .15 .8 .8]);
LastTrialEncoderPlot(BpodSystem.GUIHandles.EncoderAxes, 'init', S.threshold);

%% Setup stepper module
BpodSystem.PluginObjects.S.holdRMScurrent = 200; % Immobilize lever

%% Setup rotary encoder module
BpodSystem.PluginObjects.R.setPosition(0); % Set the current position to equal 0
BpodSystem.PluginObjects.R.thresholds = S.threshold;
BpodSystem.PluginObjects.R.sendThresholdEvents = 'on'; % If on, rotary encoder module will send threshold events to state machine
BpodSystem.PluginObjects.R.startUSBStream;

BpodSystem.SoftCodeHandlerFunction = 'SoftCodeHandler_LeverReset';

%% Main trial loop
MaxTrials = 10000;
for currentTrial = 1:MaxTrials
    S.GUI.currentTrial = currentTrial; % This is pushed out to the GUI in the next line
    S = BpodParameterGUI('sync', S); % Sync parameters with BpodParameterGUI plugin
    sma = NewStateMachine(); % Initialize new state machine description
    sma = AddState(sma, 'Name', 'WaitForPress1', ...
        'Timer', 0,...
        'StateChangeConditions', {'RotaryEncoder1_1', 'LeverRetract1'},...
        'OutputActions', {'Stepper1', ['i' 0 0], 'RotaryEncoder1', ['E#' 0], 'PWM1', 255}); % ['i' 0 0] turns off the stepper holding current. 
                                                                                       % 'E' enables the rotary encoder thresholds
                                                                                       % ['#' 0] sends byte 0x0 to the rotary encoder module for 
                                                                                       %         a trial start timestamp on its clock 
                                                                                       % 'PWM1', 255 sets Port1 LED to max brightness
    sma = AddState(sma, 'Name', 'LeverRetract1', ...
        'Timer', 0,...
        'StateChangeConditions', {'SoftCode1', 'WaitForPress2'},... % When the PC is done resetting the lever, it sends soft code 1 to the state machine
        'OutputActions', {'SoftCode', 1}); % On entering the LeverRetract state, send soft code 1 to the PC. The soft code handler will then start resetting the lever.
    sma = AddState(sma, 'Name', 'WaitForPress2', ...
        'Timer', 0,...
        'StateChangeConditions', {'RotaryEncoder1_1', 'LeverRetract2'},...
        'OutputActions', {'Stepper1', ['i' 0 0], 'RotaryEncoder1', 'E', 'PWM1', 255}); % Port1 LED turns on to indicate that the lever is ready to press
    sma = AddState(sma, 'Name', 'LeverRetract2', ...
        'Timer', 0,...
        'StateChangeConditions', {'SoftCode1', 'exit'},...
        'OutputActions', {'SoftCode', 1});
    SendStateMachine(sma);
    RawEvents = RunStateMachine;
    if ~isempty(fieldnames(RawEvents)) % If trial data was returned
        BpodSystem.Data = AddTrialEvents(BpodSystem.Data,RawEvents); % Computes trial events from raw data
        BpodSystem.Data.EncoderData{currentTrial} = BpodSystem.PluginObjects.R.readUSBStream(); % Get rotary encoder data captured since last call to R.readUSBStream()

        % Align this trial's rotary encoder timestamps to state machine trial-start (timestamp of '#' command sent from state machine to encoder module in 'TrialStart' state)
        BpodSystem.Data.EncoderData{currentTrial}.Times = BpodSystem.Data.EncoderData{currentTrial}.Times - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align timestamps to state machine's trial time 0
        BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps = BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps - BpodSystem.Data.EncoderData{currentTrial}.EventTimestamps(1); % Align event timestamps to state machine's trial time 0
        
        % Update rotary encoder plot
        TrialDuration = BpodSystem.Data.TrialEndTimestamp(currentTrial)-BpodSystem.Data.TrialStartTimestamp(currentTrial);
        LastTrialEncoderPlot(BpodSystem.GUIHandles.EncoderAxes, 'update', S.threshold, BpodSystem.Data.EncoderData{currentTrial},TrialDuration);

        SaveBpodSessionData; % Saves the field BpodSystem.Data to the current data file
    end
    HandlePauseCondition; % Checks to see if the protocol is paused. If so, waits until user resumes.
    if BpodSystem.Status.BeingUsed == 0 % If the user has pressed the 'Stop' button to end the protocol,
        BpodSystem.PluginObjects.R.stopUSBStream; % Stop streaming position data
        BpodSystem.PluginObjects.R.sendThresholdEvents = 'off'; % Stop sending threshold events to state machine
        BpodSystem.PluginObjects.S.holdRMScurrent = 0; % Release the lever
        BpodSystem.PluginObjects.R = [];
        BpodSystem.PluginObjects.S = [];
        return
    end
end


