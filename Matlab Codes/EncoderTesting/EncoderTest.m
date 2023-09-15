global BpodSystem

%% Assert Rotary Encoder module is present + USB-paired (via USB button on console GUI)
BpodSystem.assertModule('RotaryEncoder', 1);
% Create an instance of the RotaryEncoder module
R = RotaryEncoderModule(BpodSystem.ModuleUSB.RotaryEncoder1); 
% Ensure Rotary Encoder module is version 2
if BpodSystem.Modules.HWVersion_Major(strcmp(BpodSystem.Modules.Name, 'RotaryEncoder1')) < 2
    error('Error: This protocol requires rotary encoder module v2 or newer');
end

%% Define task parameters
%S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S
S = struct;
if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
    S.GUI.RewardAmount = 3; %Unit = ?l
    S.GUI.InitDelay = 1; % How long the test subject must keep the wheel motionless to receive the stimulus. Unit = seconds
    S.GUI.ResponseTime = 5; % How long until the subject must make a choice, or forefeit the trial
    S.GUI.ErrorDelay = 3; % How long the subject must wait to start the next trial after an incorrect choice
    S.GUI.InitThreshold = 3; % How much the wheel may move during the initialization period without resetting the init delay. Units = degrees
    S.GUI.ChoiceThreshold = 40; % Wheel position in the correct or incorrect direction at which a choice is registered. Units = degrees (unsigned)
    S.GUI.GaborTilt = 0; % Degrees
    S.GUI.GaborContrast = 100; % Percent of full scale
    S.GUIPanels.Reward = {'RewardAmount'}; % GUIPanels organize the parameters into groups.
    S.GUIPanels.Time = {'InitDelay', 'ResponseTime', 'ErrorDelay'};
    S.GUIPanels.Wheel = {'InitThreshold', 'ChoiceThreshold'};
    S.GUIPanels.Stimulus = {'GaborTilt', 'GaborContrast'};
end

%% Define trials
MaxTrials = 1000; % Maximum number of trials in the session. Session can be manually ended before MaxTrials from the console GUI.
TrialTypes = round(rand(1,MaxTrials)); % TrialType 0 = left correct, TrialType 1 = rightCorrect
CorrectDirection = TrialTypes; CorrectDirection(CorrectDirection == 0) = -1; % Correct response direction for each trial (-1 for left, 1 for right)
BpodSystem.Data.TrialTypes = []; % The trial type of each trial completed will be added here.

%% Initialize plots
%-- Side outcome plot (a plugin included in the Bpod_Gen2 repository)
BpodSystem.ProtocolFigures.SideOutcomePlotFig = figure('Position', [50 540 1100 250],'name','Outcome plot','numbertitle','off', 'MenuBar', 'none', 'Resize', 'off');
BpodSystem.GUIHandles.SideOutcomePlot = axes('Position', [.08 .3 .89 .6]);
SideOutcomePlot(BpodSystem.GUIHandles.SideOutcomePlot,'init',TrialTypes);
%-- Parameter GUI
BpodParameterGUI('init', S); % Initialize parameter GUI plugin

%% Setup rotary encoder module
R.useAdvancedThresholds = 'on'; % Advanced thresholds are available on rotary encoder module r2.0 or newer.
                                % See notes in setAdvancedThresholds() function in /Modules/RotaryEncoderModule.m for parameters and usage
R.sendThresholdEvents = 'on'; % Enable sending threshold crossing events to state machine
R.userCallbackFcn = 'WheelPosCallback';
R.setAdvancedThresholds([-S.GUI.ChoiceThreshold S.GUI.ChoiceThreshold S.GUI.InitThreshold], [0 0 1], [0 0 S.GUI.InitDelay]); % Syntax: setAdvancedThresholds(thresholds, thresholdTypes, thresholdTimes)

R.push(); % Makes newly loaded thresholds current
ThresholdsEnabled = [1 1 1 0 0 0 0 0];
R.enableThresholds(ThresholdsEnabled);
%% Set up visual stimulus
ScreenID = 2; % Configure to PC
pause(1.0); % matlab seems to require a pause here before clearing screen with play(0), 
            % otherwise can get stuck on Psychtoolbox splash screen
            % might need longer delay if purple image hangs on window open

%% start encoder USB stream
%R.startUSBStream; % Begin streaming position data to PC via USB


%R.streamUI();  % GUI to show event threshold function, doesn't work with
%advanced thresholds

R.zeroPosition(); % reset encoder zero reference


%P = R.currentPosition

if 0
    R.stopUSBStream; % Stop streaming positions from rotary encoder module
    clear R;
end



