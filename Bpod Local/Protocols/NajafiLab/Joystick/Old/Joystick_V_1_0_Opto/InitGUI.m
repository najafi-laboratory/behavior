classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        S.GUI.currentTrial = 0;
        
        % joystick params
        S.GUI.Threshold = 2; % Threshold for completion of a lever press, units = degrees of shaft rotation        
        S.GUI.PressWindow_s = 1.8; % how long mouse has to press lever
        S.GUI.PressWindowExtend_s = 1; % additional time added for warmup trials
        S.GUI.Reps = 2;   % number of required repeated lever presses in a trial
        S.GUIMeta.Reps.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.Reps.String = {'1', '2', '3'};
        S.GUI.ZeroRTrials = 1;
        S.GUI.ResistanceLevel = 1;
        S.GUIMeta.ResistanceLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.ResistanceLevel.String = {'0 mA', '30 mA', '91 mA', '122 mA', '152 mA', '183 mA', '214 mA', '244 mA', '900 mA'};
        % S.GUI.PostRewardDelay_s = 0.500; % post reward delay prior to lever return        
        S.GUI.ServoInPos = 1604.00; % lever start pos
        S.GUI.ServoOutPos = 27; % can press lever
        S.GUI.RetractThreshold = 0.5;
        % S.GUI.Reward_Rep = 0; % reward after each press rep?
        % S.GUIMeta.Reward_Rep.Style = 'checkbox';
        % S.GUI.CenterValveTime_s = 0.06;
        % S.GUI.CenterValveTime_Rep_s = 0.05;
        % S.GUI.CenterValveAmount_uL = 1;
        % S.GUI.CenterValveAmountRep_percent = 0.5;
        S.GUI.VisStim2Enable = 1;
        S.GUIMeta.VisStim2Enable.Style = 'checkbox';
        % S.GUI.PressVisDelay_s = 2;
        S.GUI.PressVisDelayShort_s = 0;
        S.GUI.PressVisDelayLong_s = 0.050;
        S.GUI.EarlyPressThreshold = 1;
        S.GUI.SelfTimedMode = 0;
        S.GUIMeta.SelfTimedMode.Style = 'checkbox';
        S.GUI.PrePress2Delay_s = 0.050;
        S.GUI.EnableAutoDelay = 1;
        S.GUIMeta.EnableAutoDelay.Style = 'checkbox';        
        % S.GUI.AutoDelayStart_s = 0.015;
        S.GUI.AutoDelayStep_s = 0.0005;
        % S.GUI.AutoDelayStepSelf_s = 0.002;
        % S.GUI.NumDelaySteps = 0;
        S.GUI.AutoDelayMaxVis_s = 0.500;
        S.GUI.AutoDelayMaxSelf_s = 0.800;
        % S.GUI.ResetAutoDelay = 0;
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
        % S.GUIPanels.Joystick = {'Threshold', 'PressWindow_s', 'PressWindowExtend_s', 'Reps', 'ZeroRTrials', 'ResistanceLevel', 'PostRewardDelay_s', 'ServoInPos', 'ServoOutPos', 'Reward_Rep', 'CenterValveAmount_uL', 'CenterValveAmountRep_percent', 'VisStim2Enable', 'PressVisDelayShort_s', 'PressVisDelayLong_s', 'EarlyPressThreshold', 'SelfTimedMode', 'PrePress2Delay_s', 'EnableManualTrialType', 'ManualTrialType', 'TrialTypeSequence', 'NumTrialsPerBlock'};                
        % S.GUIPanels.Joystick = {'Threshold', 'PressWindow_s', 'PressWindowExtend_s', 'Reps', 'ZeroRTrials', 'ResistanceLevel', 'ServoInPos', 'ServoOutPos', 'RetractThreshold', 'VisStim2Enable', 'PressVisDelayShort_s', 'PressVisDelayLong_s', 'EarlyPressThreshold', 'SelfTimedMode', 'PrePress2Delay_s', 'EnableAutoDelay', 'AutoDelayStart_s', 'AutoDelayStep_s', 'AutoDelayMaxVis_s', 'AutoDelayMaxSelf_s', 'ResetAutoDelay', 'EnableManualTrialType', 'ManualTrialType', 'TrialTypeSequence', 'NumTrialsPerBlock'};
        S.GUIPanels.Joystick = {'Threshold', 'PressWindow_s', 'PressWindowExtend_s', 'Reps', 'ZeroRTrials', 'ResistanceLevel', 'ServoInPos', 'ServoOutPos', 'RetractThreshold', 'VisStim2Enable', 'PressVisDelayShort_s', 'PressVisDelayLong_s', 'EarlyPressThreshold', 'SelfTimedMode', 'PrePress2Delay_s', 'EnableAutoDelay', 'AutoDelayStep_s', 'AutoDelayMaxVis_s', 'AutoDelayMaxSelf_s', 'EnableManualTrialType', 'ManualTrialType', 'TrialTypeSequence', 'NumTrialsPerBlock'};

        % Optogentic params
        S.GUI.SessionType = 2;
        S.GUIMeta.SessionType.Style = 'popupmenu';
        S.GUIMeta.SessionType.String = {'Opto', 'Control'};
        S.GUI.PulseType = 1;
        S.GUIMeta.PulseType.Style = 'popupmenu';
        S.GUIMeta.PulseType.String = {'On', 'Square', 'Sinusoidal'};
        S.GUI.PulseFreq_Hz = 50;
        S.GUI.PulseOnDur_ms = 5;
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
        S.GUI.OnFraction = 0.9;
        S.GUI.NumOptoTrialsPerBlock = 50;          
        S.GUIPanels.Opto = {'SessionType', 'PulseType', 'PulseFreq_Hz', 'PulseOnDur_ms', 'OptoVis1', 'OptoWaitForPress1', 'OptoVis2', 'OptoWaitForPress2', 'OptoTrialTypeSeq', 'OnFraction', 'NumOptoTrialsPerBlock'};

        % reward
        S.GUI.Reward_Rep = 0; % reward after each press rep?
        S.GUIMeta.Reward_Rep.Style = 'checkbox';
        S.GUI.PostRewardDelay_s = 0.500; % post reward delay prior to lever return
        S.GUI.CenterValveAmount_uL = 1;
        S.GUI.CenterValveAmountRep_percent = 0.5;
        S.GUIPanels.Reward = {'Reward_Rep', 'PostRewardDelay_s', 'CenterValveAmount_uL', 'CenterValveAmountRep_percent'};        
        
        % ITI params
        S.GUI.SetManualITI = 0;
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
        % S.GUI.ManuallTimeOutPunish = 0;
        % S.GUIMeta.ManuallTimeOutPunish.Style = 'checkbox';
        % S.GUI.TimeOutPunishMin = 1;
        % S.GUI.TimeOutPunishMax = 5;
        % S.GUI.TimeOutPunishMean = 3;
        % S.GUIPanels.ITI_Dist = {'SetManualITI', 'ForceITIZero', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'ManuallTimeOutPunish', 'TimeOutPunishMin', 'TimeOutPunishMax', 'TimeOutPunishMean'};
        S.GUIPanels.ITI_Dist = {'SetManualITI', 'ManualITI', 'ForceITIZero', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'TimeOutPunish'};
    
        % init cue params
              
        % training level params
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 5;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.TrainingLevel.String = {'Habituation', 'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumEasyWarmupTrials = 20;
        % S.GUI.WaitDurOrig_s = 0.0; % gui shows PrePertubDur as the default value for wait_dur_orig, because if mouse side licks before this time, it must be all chance, so we want wait_dur to be at least PrePerturbDur
        % S.GUI.WaitDurStep_s = 0.01; % per non early-choice trial, add this much to the original waitDur (ie the dur during the vis stim that the mouse is not allowed to sidelick)
        %S.GUIPanels.Training = {'TrainingLevel', 'NumEasyWarmupTrials', 'WaitDurOrig_s', 'WaitDurStep_s'};
        S.GUIPanels.Training = {'MaxTrials', 'TrainingLevel', 'NumEasyWarmupTrials'};
    
        % difficulty params       
    
        % audio stim
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1;  % volume control
        %S.GUI.AudioStimFreq_Hz = 15000; % Frequency of audio stim
        S.GUI.AudioStimFreq_Hz = 11025; % Frequency of audio stim, even multiple of SF = 44100
        S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};
    
        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.1; % Duration of grating stimulus in seconds - UPDATE
        % S.GUI.GratingDur_s = 1; % Duration of grating stimulus in seconds - UPDATE
        S.GUI.ISIOrig_s = 0.5; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
        S.GUI.ISIOrig_s = 1; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
        % S.GUI.NumISIOrigRep = 2; % number of grating/gray repetitions for vis stim first segment prior to perturbation
        % S.GUI.ExtraStimDurPostRew_Naive_s = 5; % naive mouse sees stimulus for this time (sec) after correct lick    
        % S.GUI.PostPerturbDurMultiplier = 2; % scaling factor for post perturbation stimulus (postperturb = preperturb * PostPerturbDurMultiplier)    
        % S.GUI.MinISIPerturb_ms = 100; % min time in ms for perturbation range from grating
        S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
        S.GUI.PreGoCueDelay_s = 0;
        % S.GUI.EasyMax = 1;
        % S.GUIMeta.EasyMax.Style = 'popupmenu';
        % S.GUIMeta.EasyMax.String = {'Default', 'Activated', 'Deactivated'};
        % S.GUIPanels.VisStim = {'EasyMax', 'VisStimEnable', 'GratingDur_s', 'ISIOrig_s', 'NumISIOrigRep', 'PostPerturbDurMultiplier', 'ExtraStimDurPostRew_Naive_s', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s'}; 
        S.GUIPanels.VisStim = {'VisStimEnable', 'GratingDur_s', 'ISIOrig_s', 'PreVisStimDelay_s', 'PreGoCueDelay_s'};         

        % contingency and bias params
        % S.GUI.ShortISIFraction = 0.5;   % set fraction of trials that are short ISI (long ISI fraction = (1 - short))
        % S.GUI.RepeatedIncorrect = 0;
        % S.GUIMeta.RepeatedIncorrect.Style = 'checkbox';
        % S.GUI.RepeatedProb = 1.0;
        % S.GUI.AdjustValve = 1;
        % S.GUIMeta.AdjustValve.Style = 'checkbox';
        % S.GUI.AdjustValveThres = 3;
        % S.GUI.AdjustValveTime = 0.02;
        % S.GUI.AdjustFraction = 0.65;
        % S.GUIPanels.Contingency_Bias = {'ShortISIFraction', 'RepeatedIncorrect', 'RepeatedProb', 'AdjustValve', 'AdjustValveThres', 'AdjustValveTime', 'AdjustFraction'};
    
        % reward params
        
        % S.GUI.LeftValveTime_s = 0.15;
        % S.GUI.RightValveTime_s = 0.15;
        % 
        % S.GUI.WindowRewardGrabDuration_Naive_s = 10;  % naive mouse has up to x seconds to grab reward    
        % S.GUI.RewardDelay_s = 0; % How long the mouse must wait in the goal port for reward to be delivered
        % S.GUIMeta.EnableCenterLick_Trained.Style = 'checkbox';
        % S.GUI.WindCenterLick_s = 2;   
        % S.GUIPanels.Reward = {'LeftValveTime_s', 'RightValveTime_s', 'WindowRewardGrabDuration_Naive_s', 'RewardDelay_s', 'WindCenterLick_s'};


    
        % punish params
        S.GUI.IncorrectSoundVolume_percent = 0.07;  % volume control
        S.GUI.PunishSoundDuration_s = 1; % Seconds to wait on errors before next trial can start
        S.GUI.IncorrectSound = 1; % if 1, plays a white noise pulse on error. if 0, no sound is played.
        S.GUIMeta.IncorrectSound.Style = 'checkbox';
        S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound'}; 
    
        % choice params
        % S.GUI.ChoiceWindow_s = 5; % How long after go cue until the mouse must make a choice
        % S.GUI.ConfirmLickInterval_s = 0.2; % min interval until choice can be confirmed    
        % S.GUI.ChoiceConfirmWindow_s = 5; % time during which correct choice can be confirmed    
        % S.GUIPanels.Choice = {'ChoiceWindow_s', 'ConfirmLickInterval_s', 'ChoiceConfirmWindow_s'};

    end
end















    end
end
