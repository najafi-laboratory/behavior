classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        S.GUI.currentTrial = 0;
        
        % joystick params
        S.GUI.ChemogeneticSession = 0;
        S.GUIMeta.ChemogeneticSession.Style = 'checkbox';
        S.GUI.mgCNO = '1';
        S.GUI.mlSaline = '1';
        S.GUI.Threshold = 2; % Threshold for completion of a lever press, units = degrees of shaft rotation        
        S.GUI.WarmupThreshold = S.GUI.Threshold / 2; % Threshold for completion of a lever press, units = degrees of shaft rotation        
        S.GUI.Press1Window_s = 1.8; % how long mouse has to press lever
        S.GUI.Press2Window_s = 1.8; % how long mouse has to press lever
        S.GUI.PressWindowExtend_s = 1; % additional time added for warmup trials
        S.GUI.Reps = 2;   % number of required repeated lever presses in a trial
        S.GUIMeta.Reps.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.Reps.String = {'1', '2', '3'};
        S.GUI.ZeroRTrials = 1;
        S.GUI.ResistanceLevel = 1;
        S.GUIMeta.ResistanceLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.ResistanceLevel.String = {'0 mA', '30 mA', '91 mA', '122 mA', '152 mA', '183 mA', '214 mA', '244 mA', '900 mA'};
        % set servo positions per rig, allows consistent code version
        % updates
        switch BpodSystem.Data.RigName
            case 'ImagingRig'
                S.GUI.ServoInPos = 1570.00; % lever start pos
                S.GUI.ServoOutPos = 34; % can press lever
            case 'JoystickRig'
                S.GUI.ServoInPos = 1598.00; % lever start pos
                S.GUI.ServoOutPos = 34; % can press lever
            case 'JoystickRig2'
                S.GUI.ServoInPos = 1598.00; % lever start pos
                S.GUI.ServoOutPos = 34; % can press lever            
        end
        S.GUI.RetractThreshold = 0.3;
        S.GUI.EarlyPressThreshold = 1;
        S.GUI.SelfTimedMode = 0;
        S.GUIMeta.SelfTimedMode.Style = 'checkbox';
        S.GUI.PrePress2DelayShort_s = 0.030;
        S.GUI.PrePress2DelayLong_s = 0.050;        
        S.GUI.EnableAutoDelay = 1;
        S.GUIMeta.EnableAutoDelay.Style = 'checkbox';        
        S.GUI.AutoDelayStep_s = 0.0003;
        S.GUI.AutoDelayMaxShort_s = 0.200;
        S.GUI.AutoDelayMaxLong_s = 0.500;
        S.GUI.EnableManualTrialType = 0;
        S.GUIMeta.EnableManualTrialType.Style = 'checkbox';
        S.GUI.ManualTrialType = 1;
        S.GUIMeta.ManualTrialType.Style = 'popupmenu';
        S.GUIMeta.ManualTrialType.String = {'Short', 'Long'};
        S.GUI.TrialTypeSequence = 3;
        S.GUIMeta.TrialTypeSequence.Style = 'popupmenu';
        S.GUIMeta.TrialTypeSequence.String = {'Random', 'Random First Block', 'Short First Block', 'Long First Block'};
        S.GUI.NumTrialsPerBlock = 50;
        S.GUI.BlockLengthMargin = 5;
        S.GUI.ProbeTrialFraction = 0.1;
        S.GUIPanels.Joystick = {'ChemogeneticSession', 'mgCNO', 'mlSaline', 'Threshold', 'WarmupThreshold', 'Press1Window_s', 'Press2Window_s', 'PressWindowExtend_s', 'Reps', 'ZeroRTrials', 'ResistanceLevel', 'ServoInPos', 'ServoOutPos', 'RetractThreshold', 'EarlyPressThreshold', 'SelfTimedMode', 'PrePress2DelayShort_s', 'PrePress2DelayLong_s', 'EnableAutoDelay', 'AutoDelayStep_s', 'AutoDelayMaxShort_s', 'AutoDelayMaxLong_s', 'EnableManualTrialType', 'ManualTrialType', 'TrialTypeSequence', 'NumTrialsPerBlock', 'BlockLengthMargin', 'ProbeTrialFraction'};

        % assisted trials
        S.GUI.AssistedTrials = @(src,event)fnAssistedTrials;
        S.GUIMeta.AssistedTrials.Style = 'pushbutton';
        S.GUI.ATRangeStart = 0;
        S.GUI.ATRangeStop = 0;
        S.GUIPanels.AssistedTrials = {'AssistedTrials', 'ATRangeStart', 'ATRangeStop'};

        % Optogentic params
        S.GUI.SessionType = 2;  % S.GUI.SessionType = 2;
        S.GUIMeta.SessionType.Style = 'popupmenu';
        S.GUIMeta.SessionType.String = {'Opto', 'Control'};
        S.GUI.PulseType = 3;
        S.GUIMeta.PulseType.Style = 'popupmenu';
        S.GUIMeta.PulseType.String = {'ContinuousOpto', 'SquareWaveOpto_ContinuousShutter', 'SquareWaveOpto_PulsedShutter'};
        S.GUI.MaxOptoDur_s = 2.5;
        S.GUI.LEDOnPulseDur_ms = 7.8;
        S.GUI.LEDOffPulseDur_ms = 92.2;
        S.GUI.OptoVis1 = 1;
        S.GUIMeta.OptoVis1.Style = 'checkbox';
        S.GUI.OptoWaitForPress1 = 1;
        S.GUIMeta.OptoWaitForPress1.Style = 'checkbox';
        S.GUI.OptoVis2 = 1;
        S.GUIMeta.OptoVis2.Style = 'checkbox';
        S.GUI.OptoWaitForPress2 = 1;
        S.GUIMeta.OptoWaitForPress2.Style = 'checkbox';
        S.GUI.OptoTrialTypeSeq = 5;
        S.GUIMeta.OptoTrialTypeSeq.Style = 'popupmenu';
        S.GUIMeta.OptoTrialTypeSeq.String = {'Random', 'Random First Block', 'Off First Block', 'On First Block', 'On Epoch'};               
        S.GUI.OnFraction = 0.35;    % S.GUI.OnFraction = 0.35;
        S.GUI.NumOptoTrialsPerBlock = 50; 
        S.GUI.EpochTrialStart = 1;
        S.GUI.EpochTrialStop = 15;        
        S.GUIPanels.Opto = {'SessionType', 'PulseType', 'MaxOptoDur_s', 'LEDOnPulseDur_ms', 'LEDOffPulseDur_ms','OptoVis1', 'OptoWaitForPress1', 'OptoVis2', 'OptoWaitForPress2', 'OptoTrialTypeSeq', 'OnFraction', 'NumOptoTrialsPerBlock', 'EpochTrialStart', 'EpochTrialStop'};

        % reward
        S.GUI.EnableAutoPreRewardDelay = 1;
        S.GUIMeta.EnableAutoPreRewardDelay.Style = 'checkbox';                
        S.GUI.AutoPreRewardDelayStep_s = 0.0001;
        S.GUI.AutoPreRewardDelayMax_s = 0.100;        
        S.GUI.PreRewardDelay_s = 0.030;
        S.GUI.PostRewardDelay_s = 0.500; % post reward delay prior to lever return
        S.GUI.CenterValveAmount_uL = 1;
        S.GUIPanels.Reward = {'EnableAutoPreRewardDelay', 'AutoPreRewardDelayStep_s', 'AutoPreRewardDelayMax_s', 'PreRewardDelay_s', 'PostRewardDelay_s', 'CenterValveAmount_uL'};        
        
        % ITI params
        S.GUI.ITI_Post = 0.500;  % V_3_3 ITI post, fixed duration
        S.GUI.SetManualITI = 0; % S.GUI.SetManualITI = 0;
        S.GUIMeta.SetManualITI.Style = 'checkbox';
        S.GUI.ManualITI = '0';
        S.GUI.ITIMin = 1;    % Minimum ITI (in seconds)
        S.GUI.ITIMax = 5;    % Maximum ITI (in seconds)
        S.GUI.ITIMean = 3;   % Mean ITI (in seconds)
        S.GUI.PunishITI = 1;  % early press and didnt press punish ITI (from EarlyPress1Punish, EarlyPress2Punish, Punish)
        S.GUIPanels.ITI_Dist = {'ITI_Post', 'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'PunishITI'};
    
        % init cue params
              
        % training level params
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 5;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.TrainingLevel.String = {'Habituation', 'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumEasyWarmupTrials = 10;
        S.GUIPanels.Training = {'MaxTrials', 'TrainingLevel', 'NumEasyWarmupTrials'};
    
        % difficulty params       
    
        % audio stim
        S.GUI.AudioStimEnable = 0;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1;  % volume control
        S.GUI.AudioStimFreq_Hz = 11025; % Frequency of audio stim, even multiple of SF = 44100
        S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};
    
        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.1; % Duration of grating stimulus in seconds - UPDATE
        S.GUI.ISIOrig_s = 0.5; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
        S.GUI.ISIOrig_s = 1; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
        S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
        S.GUI.PreGoCueDelay_s = 0;
        S.GUIPanels.VisStim = {'VisStimEnable', 'GratingDur_s', 'ISIOrig_s', 'PreVisStimDelay_s', 'PreGoCueDelay_s'};         

        % punish params
        S.GUI.IncorrectSoundVolume_percent = 0.07;  % volume control
        S.GUI.PunishSoundDuration_s = 1; % Seconds to wait on errors before next trial can start
        S.GUI.IncorrectSound = 1; % if 1, plays a white noise pulse on error. if 0, no sound is played.
        S.GUIMeta.IncorrectSound.Style = 'checkbox';
        S.GUI.EarlyPressPunishSoundVolume_percent = 0.00;
        S.GUI.EarlyPressPunishSoundDuration_s = 1;
        

        S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound', 'EarlyPressPunishSoundVolume_percent', 'EarlyPressPunishSoundDuration_s'}; 
    
    end
end
    end
end
