classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        S.GUI.currentTrial = 0;
        
        % joystick params
        S.GUI.ChemogeneticSession = 0;
        S.GUIMeta.ChemogeneticSession.Style = 'checkbox';
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
                S.GUI.ServoInPos = 1601.00; % lever start pos
                S.GUI.ServoOutPos = 34; % can press lever
        end
        S.GUI.RetractThreshold = 0.3;
        S.GUI.PreVis2DelayShort_s = 0;
        S.GUI.PreVis2DelayLong_s = 0.050;
        S.GUI.EarlyPressThreshold = 1;
        S.GUI.SelfTimedMode = 0;
        S.GUIMeta.SelfTimedMode.Style = 'checkbox';
        S.GUI.PrePress2DelayShort_s = 0.010;    % added version 1_5_1, self-timed now has short/long blocks like vis guided
        S.GUI.PrePress2DelayLong_s = 0.500;
        S.GUI.EnableAutoDelay = 1;
        S.GUIMeta.EnableAutoDelay.Style = 'checkbox';        
        S.GUI.AutoDelayStep_s = 0.0001;
        S.GUI.AutoDelayMaxVis_s = 0.500;
        S.GUI.AutoDelayMaxSelf_s = 0.800;
        S.GUIMeta.ResetAutoDelay.Style = 'checkbox';        
        S.GUI.EnableManualTrialType = 0;
        S.GUIMeta.EnableManualTrialType.Style = 'checkbox';
        S.GUI.ManualTrialType = 1;
        S.GUIMeta.ManualTrialType.Style = 'popupmenu';
        S.GUIMeta.ManualTrialType.String = {'Short', 'Long'};
        S.GUI.TrialTypeSequence = 3;
        S.GUIMeta.TrialTypeSequence.Style = 'popupmenu';
        S.GUIMeta.TrialTypeSequence.String = {'Random', 'Random First Block', 'Short First Block', 'Long First Block'};
        S.GUI.NumTrialsPerBlock = 50;        
        S.GUIPanels.Joystick = {'ChemogeneticSession', 'Threshold', 'WarmupThreshold', 'Press1Window_s', 'Press2Window_s', 'PressWindowExtend_s', 'Reps', 'ZeroRTrials', 'ResistanceLevel', 'ServoInPos', 'ServoOutPos', 'RetractThreshold', 'PreVis2DelayShort_s', 'PreVis2DelayLong_s', 'EarlyPressThreshold', 'SelfTimedMode', 'PrePress2DelayShort_s', 'PrePress2DelayLong_s', 'EnableAutoDelay', 'AutoDelayStep_s', 'AutoDelayMaxVis_s', 'AutoDelayMaxSelf_s', 'EnableManualTrialType', 'ManualTrialType', 'TrialTypeSequence', 'NumTrialsPerBlock'};

        % Optogentic params
        S.GUI.SessionType = 1;  % S.GUI.SessionType = 2;
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
        S.GUI.OptoTrialTypeSeq = 1;
        S.GUIMeta.OptoTrialTypeSeq.Style = 'popupmenu';
        S.GUIMeta.OptoTrialTypeSeq.String = {'Random', 'Random First Block', 'Off First Block', 'On First Block'};
        S.GUI.OnFraction = 1;    % S.GUI.OnFraction = 0.35;
        S.GUI.NumOptoTrialsPerBlock = 50;          
        S.GUIPanels.Opto = {'SessionType', 'PulseType', 'MaxOptoDur_s', 'LEDOnPulseDur_ms', 'LEDOffPulseDur_ms','OptoVis1', 'OptoWaitForPress1', 'OptoVis2', 'OptoWaitForPress2', 'OptoTrialTypeSeq', 'OnFraction', 'NumOptoTrialsPerBlock'};

        % reward
        S.GUI.Reward_Rep = 0; % reward after each press rep?
        S.GUIMeta.Reward_Rep.Style = 'checkbox';
        S.GUI.PostRewardDelay_s = 0.500; % post reward delay prior to lever return
        S.GUI.CenterValveAmount_uL = 1;
        S.GUI.CenterValveAmountRep_percent = 0.5;
        S.GUIPanels.Reward = {'Reward_Rep', 'PostRewardDelay_s', 'CenterValveAmount_uL', 'CenterValveAmountRep_percent'};        
        
        % ITI params
        S.GUI.SetManualITI = 1; % S.GUI.SetManualITI = 0;
        S.GUIMeta.SetManualITI.Style = 'checkbox';
        S.GUI.ManualITI = '0';
        S.GUI.ForceITIZero = 0;
        S.GUIMeta.ForceITIZero.Style = 'checkbox';
        S.GUI.ITIMin = 1;    % Minimum ITI (in seconds)
        S.GUI.ITIMax = 5;    % Maximum ITI (in seconds)
        S.GUI.ITIMean = 3;
        S.GUI.ActTimeOutPunish = 1;
        S.GUIMeta.ActTimeOutPunish.Style = 'checkbox';
        S.GUI.TimeOutPunish = 2;
        S.GUIPanels.ITI_Dist = {'SetManualITI', 'ManualITI', 'ForceITIZero', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'TimeOutPunish'};
    
        % init cue params
              
        % training level params
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 5;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.TrainingLevel.String = {'Habituation', 'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumEasyWarmupTrials = 20;
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
        S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound'}; 
    
    end
end















    end
end
