classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        S.GUI.currentTrial = 0;
        
        % joystick params
        S.GUI.ExperimenterInitials = 'Initials';
        S.GUI.SelfTimedMode = 1;
        S.GUIMeta.SelfTimedMode.Style = 'checkbox';  
        S.GUI.ChemogeneticSession = 0;
        S.GUIMeta.ChemogeneticSession.Style = 'checkbox';

        S.GUI.Threshold = 0.7; % Threshold for completion of a lever press, units = degrees of shaft rotation
        S.GUI.EarlyPressThreshold = 0.7;
        S.GUI.WarmupThreshold = 0.7;
        S.GUI.RetractThreshold = 0.3;   
        % window setting parameters 
        S.GUI.Press1Window_s = 3; % how long mouse has to press lever
        S.GUI.Press2WindowShort_s = 2; % how long mouse has to press lever 
        S.GUI.Press2WindowLong_s = 2.5; % how long mouse has to press lever
        S.GUI.Press2WindowWarmup_s = 5;
        S.GUI.EnableAutoPressWinReduce = 0;
        S.GUIMeta.EnableAutoPressWinReduce.Style = 'checkbox'; 
        S.GUI.AutoPressWin1ReduceMin = 0.500;
        S.GUI.AutoPressWinShortReduceMin = 0.500;
        S.GUI.AutoPressWinLongReduceMin = 0.500;
        S.GUI.AutoPressWinReduceStep = 0.001;
    
        S.GUI.ServoInPos = 1678.00; % lever start pos            
        S.GUI.ServoOutPos = 34; % can press lever
        
        S.GUI.PrePress2DelayShort_s = 0.400;
        S.GUI.PrePress2DelayLong_s = 0.800;

        S.GUI.EnableAutoDelay = 0;
        S.GUIMeta.EnableAutoDelay.Style = 'checkbox';                
        S.GUI.AutoDelayMaxShort_s = 0.300;
        S.GUI.AutoDelayMaxLong_s = 1.000;
        S.GUI.AutoDelayStep_s = 0.0003;
        S.GUI.AssistMode = 1;
        S.GUIMeta.AssistMode.Style = 'checkbox';
        S.GUI.AssistProb = 0.3;

        S.GUI.EnableManualTrialType = 0;
        S.GUIMeta.EnableManualTrialType.Style = 'checkbox';
        S.GUI.ManualTrialType = 1;
        S.GUIMeta.ManualTrialType.Style = 'popupmenu';
        S.GUIMeta.ManualTrialType.String = {'Short', 'Long'};
        % block setting parameters
        S.GUI.EnableBlockChange = 1;
        S.GUIMeta.EnableBlockChange.Style = 'checkbox';
        S.GUI.EnableSingleBlock = 0;
        S.GUIMeta.EnableSingleBlock.Style = 'checkbox';
        S.GUI.EnableProbe = 0;
        S.GUIMeta.EnableProbe.Style = 'checkbox';
        S.GUI.TrialTypeSequence = 4;
        S.GUIMeta.TrialTypeSequence.Style = 'popupmenu';
        S.GUIMeta.TrialTypeSequence.String = {'Random', 'Random First Block', 'Short First Block', 'Long First Block'};
        S.GUI.NumTrialsPerBlock = 35;
        S.GUI.MaxNumTrialsPerBlock = 50;
        S.GUI.BlockLengthMargin = 5;
        S.GUI.MinJitterRange = 5;
        S.GUI.MaxJitterRange = 10;
        S.GUI.ProbeTrialFraction = 0.1;
        S.GUI.FirstEpochShortLen = 5;
        S.GUI.FirstEpochLongLen = 7;

        S.GUIPanels.Joystick = {'ExperimenterInitials', 'SelfTimedMode', 'ChemogeneticSession', 'Threshold', 'EarlyPressThreshold', 'WarmupThreshold', 'RetractThreshold', 'ServoInPos', 'ServoOutPos', 'PrePress2DelayShort_s', 'PrePress2DelayLong_s', 'EnableAutoDelay', 'AutoDelayMaxShort_s', 'AutoDelayMaxLong_s', 'AutoDelayStep_s', 'AssistMode', 'AssistProb', 'EnableManualTrialType', 'ManualTrialType'};

        % Optogentic params
        S.GUI.SessionType = 2;
        S.GUIMeta.SessionType.Style = 'popupmenu';
        S.GUIMeta.SessionType.String = {'Opto', 'Control'};
        S.GUI.PulseType = 3;
        S.GUIMeta.PulseType.Style = 'popupmenu';
        S.GUIMeta.PulseType.String = {'ContinuousOpto', 'SquareWaveOpto_ContinuousShutter', 'SquareWaveOpto_PulsedShutter', 'Continuous_RampDown'};
        S.GUI.MaxOptoDur_s = 2.5;
        S.GUI.LEDOnPulseDur_ms = 7.8;
        S.GUI.LEDOffPulseDur_ms = 92.2;
        S.GUI.OptoVis1 = 0;
        S.GUIMeta.OptoVis1.Style = 'checkbox';

        S.GUI.OptoVis2 = 0;
        S.GUIMeta.OptoVis2.Style = 'checkbox';

        S.GUI.OptoPrePressDelay = 0;
        S.GUIMeta.OptoPrePressDelay.Style = 'checkbox';

        S.GUI.OptoWaitForPress1 = 0;
        S.GUIMeta.OptoWaitForPress1.Style = 'checkbox';


        S.GUI.OptoWaitForPress2 = 0;
        S.GUIMeta.OptoWaitForPress2.Style = 'checkbox';        

        S.GUI.OptoPress1 = 0;
        S.GUIMeta.OptoPress1.Style = 'checkbox';

        S.GUI.OptoPress2 = 0;
        S.GUIMeta.OptoPress2.Style = 'checkbox';

        S.GUI.OptoRewardITI = 1;
        S.GUIMeta.OptoRewardITI.Style = 'checkbox';

        S.GUI.OptoEarlyPressITI = 1;
        S.GUIMeta.OptoEarlyPressITI.Style = 'checkbox';

        S.GUI.OptoLatePressITI = 1;
        S.GUIMeta.OptoLatePressITI.Style = 'checkbox';
        

        S.GUI.OptoITICycleType = 2;
        S.GUIMeta.OptoITICycleType.Style = 'popupmenu';
        S.GUIMeta.OptoITICycleType.String = {'NumPulses', 'Duration'};

        S.GUI.OptoITINumPulses = 5;
        S.GUI.OptoITIDur_s = 1;

        S.GUI.OptoITIAlignment = 1;
        S.GUIMeta.OptoITIAlignment.Style = 'popupmenu';
        S.GUIMeta.OptoITIAlignment.String = {'StateStart', 'StateStop'};                       

        S.GUI.OptoTrialTypeSeq = 1; % S.GUI.OptoTrialTypeSeq = 5;
        S.GUIMeta.OptoTrialTypeSeq.Style = 'popupmenu';
        S.GUIMeta.OptoTrialTypeSeq.String = {'Random', 'Random First Block', 'Off First Block', 'On First Block', 'On Epoch', 'Random First Epoch'};               
        S.GUI.OnFraction = 0.35;    % S.GUI.OnFraction = 0.35;         
        S.GUI.EpochTrialStart = 1;
        S.GUI.EpochTrialStop = 15;
        S.GUI.NumOptoTrialsPerBlock = 50;
        S.GUIPanels.Opto = {'SessionType', 'PulseType', 'MaxOptoDur_s', 'LEDOnPulseDur_ms', 'LEDOffPulseDur_ms','OptoVis1',...
            'OptoVis2', 'OptoPrePressDelay', 'OptoWaitForPress1', 'OptoWaitForPress2', 'OptoPress1', 'OptoPress2', 'OptoRewardITI',...
            'OptoEarlyPressITI', 'OptoLatePressITI', 'OptoITICycleType', 'OptoITINumPulses', 'OptoITIDur_s', 'OptoITIAlignment',...
            'OptoTrialTypeSeq', 'OnFraction', 'EpochTrialStart', 'EpochTrialStop', 'NumOptoTrialsPerBlock'};


        % reward
        S.GUI.PreRewardDelay_s = 0.1;
        S.GUI.EnableAutoPreRewardDelay = 0;
        S.GUIMeta.EnableAutoPreRewardDelay.Style = 'checkbox';                        
        S.GUI.AutoPreRewardDelayMax_s = 0.200;
        S.GUI.AutoPreRewardDelayStep_s = 0.0001;
        S.GUI.PostRewardDelay_s = 1.000; % post reward delay prior to lever return
        S.GUI.CenterValveAmount_uL = 3;
        S.GUI.EnableAlternatingReward = 0;
        S.GUIMeta.EnableAlternatingReward.Style = 'checkbox';
        S.GUI.ShortRewardAmount_uL = 1;
        S.GUI.LongRewardAmount_uL = 2;
        S.GUIPanels.Reward = {'PreRewardDelay_s', 'EnableAutoPreRewardDelay', 'AutoPreRewardDelayMax_s', 'AutoPreRewardDelayStep_s', 'PostRewardDelay_s', 'CenterValveAmount_uL' , 'EnableAlternatingReward' , 'ShortRewardAmount_uL' , 'LongRewardAmount_uL'};        
        

        % ITI params
        S.GUI.ITI_Pre = 0.500;  % V_3_3 ITI_post, fixed duration, now ITI_Pre V_3_7
        S.GUI.SetManualITI = 0; % S.GUI.SetManualITI = 0;
        S.GUIMeta.SetManualITI.Style = 'checkbox';
        S.GUI.ManualITI = '0';
        S.GUI.ITIMin = 3;    % Minimum ITI (in seconds)
        S.GUI.ITIMax = 5;    % Maximum ITI (in seconds)
        S.GUI.ITIMean = 4;   % Mean ITI (in seconds)
        S.GUI.PunishITI = 5;  % early press and didnt press punish ITI (from EarlyPress1Punish, EarlyPress2Punish, Punish)
        S.GUIPanels.ITI_Dist = {'ITI_Pre', 'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'PunishITI'};

        % init cue params
                      
        S.GUIPanels.WindowSetting = {'Press1Window_s', 'Press2WindowWarmup_s','Press2WindowShort_s','Press2WindowLong_s', ...
            'EnableAutoPressWinReduce','AutoPressWin1ReduceMin' ,'AutoPressWinShortReduceMin','AutoPressWinLongReduceMin', ...
            'AutoPressWinReduceStep'} ;


        % training level params
        S.GUI.MaxTrials = 1000;
        S.GUI.NumEasyWarmupTrials = 5;
    
        % audio stim
        S.GUI.AudioStimEnable = 0;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1;  % volume control
        S.GUI.AudioStimFreq_Hz = 11025; % Frequency of audio stim, even multiple of SF = 44100
        S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};
    
        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.1;
        S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
        S.GUIPanels.VisStim = {'VisStimEnable', 'GratingDur_s', 'PreVisStimDelay_s'};         

        % punish params
        S.GUI.IncorrectSoundVolume_percent = 0.07;  % volume control
        S.GUI.PunishSoundDuration_s = 0.1; % Seconds to wait on errors before next trial can start
        S.GUI.IncorrectSound = 0; % if 1, plays a white noise pulse on error. if 0, no sound is played.
        S.GUIMeta.IncorrectSound.Style = 'checkbox';
        S.GUI.EarlyPressPunishSoundVolume_percent = 0.00;
        S.GUI.EarlyPressPunishSoundDuration_s = 1;
        

        S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound', 'EarlyPressPunishSoundVolume_percent', 'EarlyPressPunishSoundDuration_s'}; 
        S.GUIPanels.BlockSetting = {'MaxTrials', 'NumEasyWarmupTrials', 'EnableSingleBlock','EnableBlockChange', ...
             'TrialTypeSequence','NumTrialsPerBlock','MaxNumTrialsPerBlock',...
            'BlockLengthMargin','MinJitterRange','MaxJitterRange', 'EnableProbe','ProbeTrialFraction', 'FirstEpochShortLen', 'FirstEpochLongLen'};
    end
end
    end
end
