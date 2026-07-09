classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings; % Load settings chosen in launch manager into current workspace as a struct called S

    if isempty(fieldnames(S))  % If settings file was an empty struct, populate struct with default settings
        S.GUI.GratingDur_s = 0.1; % Duration of grating stimulus in seconds - UPDATE
        S.GUI.ISIOrig_s = 0.5; % Duration of *fixed* gray screen stimulus in seconds - UPDATE
        S.GUI.NumISIOrigRep = 2; % number of grating/gray repetitions for vis stim first segment prior to perturbation
        
        % passive
        S.GUI.EnablePassive = 0;
        S.GUIMeta.EnablePassive.Style = 'checkbox';
        S.GUI.TrialPerBlock = 5;
        S.GUI.PassivePostMultiplier = 2;
        S.GUIPanels.Passive = {'EnablePassive', 'TrialPerBlock', 'PassivePostMultiplier'};

        % Servos - spouts
        S.GUI.EnableMovingSpouts = 1;
        S.GUIMeta.EnableMovingSpouts.Style = 'checkbox';
        S.GUI.RightServoInPos = 1099.50;   % servo position coordinates as per Maestro GUI
        S.GUI.LeftServoInPos = 1921.50;
        S.GUI.ServoDeflection = -122.5;
        S.GUIPanels.Servos = {'EnableMovingSpouts', 'RightServoInPos', 'LeftServoInPos', 'ServoDeflection'};

        % ITI params
        S.GUI.ForceITIZero = 0;
        S.GUIMeta.ForceITIZero.Style = 'checkbox';
        S.GUI.ITIMin = 1;    % Minimum ITI (in seconds)
        S.GUI.ITIMax = 5;    % Maximum ITI (in seconds)
        S.GUI.ITIMean = 3;
        S.GUI.ActTimeOutPunish = 1;
        S.GUIMeta.ActTimeOutPunish.Style = 'checkbox';
        S.GUI.ManuallTimeOutPunish = 0;
        S.GUIMeta.ManuallTimeOutPunish.Style = 'checkbox';
        S.GUI.TimeOutPunishMin = 1;
        S.GUI.TimeOutPunishMax = 5;
        S.GUI.TimeOutPunishMean = 3;
        S.GUIPanels.ITI_Dist = {'ForceITIZero', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'ManuallTimeOutPunish', 'TimeOutPunishMin', 'TimeOutPunishMax', 'TimeOutPunishMean'};
    
        % init cue params
        S.GUI.InitCueVolume_percent = 0.5;  % volume control
        S.GUI.InitCueDuration_s = 0.05; % Duration of init sound
        S.GUI.InitWindowTimeout_s = 5; % How long the mouse has to initiate stimulus or miss init lick
        S.GUI.InitCueFreq_Hz = 4900; % Frequency of init cue, even multiple of 44100 SF
        S.GUIPanels.InitCue = {'InitCueVolume_percent', 'InitCueDuration_s', 'InitWindowTimeout_s', 'InitCueFreq_Hz'};
    
        % go cue params
        S.GUI.GoCueVolume_percent = 100;  % volume control
        S.GUI.GoCueDuration_s = 0.05; % Duration of go sound
        S.GUI.GoCueFreq_Hz = 11025; % Frequency of go cue, even multiple of 44100 SF
        S.GUIPanels.GoCue = {'GoCueVolume_percent', 'GoCueDuration_s', 'GoCueFreq_Hz'};
    
        % training level params
        S.GUI.TrainingLevel = 6;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu'; % the GUIMeta field is used by the ParameterGUI plugin to customize UI objects.
        S.GUIMeta.TrainingLevel.String = {'Passive', 'Habituation', 'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumEasyWarmupTrials = 20;
        S.GUI.WaitDurOrig_s = 0.0; % gui shows PrePertubDur as the default value for wait_dur_orig, because if mouse side licks before this time, it must be all chance, so we want wait_dur to be at least PrePerturbDur
        S.GUI.WaitDurStep_s = 0.01; % per non early-choice trial, add this much to the original waitDur (ie the dur during the vis stim that the mouse is not allowed to sidelick)
        S.GUIPanels.Training = {'TrainingLevel', 'NumEasyWarmupTrials', 'WaitDurOrig_s', 'WaitDurStep_s'};
    
        % difficulty params
        S.GUI.PercentEasy = 100;
        S.GUI.PercentMediumEasy = 0;
        S.GUI.PercentMediumHard = 0;
        S.GUI.PercentHard = 0;    
        S.GUIPanels.Difficulty = {'PercentEasy', 'PercentMediumEasy', 'PercentMediumHard', 'PercentHard'};
    
        % audio stim
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1;  % volume control
        S.GUI.AudioStimFreq_Hz = 14700; % Frequency of audio stim, even multiple of SF = 44100
        S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};
    
        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.ExtraStim = 1;
        S.GUIMeta.ExtraStim.Style = 'popupmenu';
        S.GUIMeta.ExtraStim.String = {'Default', 'Activated', 'Deactivated'};
        S.GUI.NumExtraStim = 90;
        S.GUI.ExtraStimDurPostRew_Naive_s = 5; % naive mouse sees stimulus for this time (sec) after correct lick    
        S.GUI.MinISIPerturb_ms = 100; % min time in ms for perturbation range from grating
        S.GUI.PreVisStimDelay_s = 0; % How long the mouse must poke in the center to activate the goal port
        S.GUI.PreGoCueDelay_s = 0;
        S.GUI.EasyMax = 1;
        S.GUIMeta.EasyMax.Style = 'popupmenu';
        S.GUIMeta.EasyMax.String = {'Default', 'Activated', 'Deactivated'};
        S.GUIPanels.VisStim = {'VisStimEnable', 'ExtraStim', 'NumExtraStim', 'GratingDur_s', 'ISIOrig_s', 'NumISIOrigRep', 'ExtraStimDurPostRew_Naive_s', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s', 'EasyMax'}; 
     
        % contingency and bias params
        S.GUI.ShortISIFraction = 0.5;   % set fraction of trials that are short ISI (long ISI fraction = (1 - short))
        S.GUI.RepeatedIncorrect = 0;
        S.GUIMeta.RepeatedIncorrect.Style = 'checkbox';
        S.GUI.RepeatedProb = 1.0;
        S.GUI.AdjustValve = 1;
        S.GUIMeta.AdjustValve.Style = 'checkbox';
        S.GUI.AdjustValveThres = 2;
        S.GUI.AdjustValvePercent = 0.12;
        S.GUI.AdjustFraction = 0.65;
        S.GUIPanels.Contingency_Bias = {'ShortISIFraction', 'RepeatedIncorrect', 'RepeatedProb', 'AdjustValve', 'AdjustValveThres', 'AdjustValvePercent', 'AdjustFraction'};
    
        % reward params
        S.GUI.LeftValveTime_s = 0.075;
        S.GUI.RightValveTime_s = 0.075;
        S.GUI.CenterValveTime_s = 0.05;
        S.GUI.WindowRewardGrabDuration_Naive_s = 10;  % naive mouse has up to x seconds to grab reward    
        S.GUI.RewardDelay_s = 0; % How long the mouse must wait in the goal port for reward to be delivered
        S.GUIMeta.EnableCenterLick_Trained.Style = 'checkbox';
        S.GUI.WindCenterLick_s = 2;   
        S.GUIPanels.Reward = {'LeftValveTime_s', 'RightValveTime_s', 'CenterValveTime_s', 'WindowRewardGrabDuration_Naive_s', 'RewardDelay_s', 'WindCenterLick_s'};
    
        % punish params
        S.GUI.IncorrectSoundVolume_percent = 0.15;  % volume control
        S.GUI.PunishSoundDuration_s = 1; % Seconds to wait on errors before next trial can start
        S.GUI.IncorrectSound = 1; % if 1, plays a white noise pulse on error. if 0, no sound is played.
        S.GUIMeta.IncorrectSound.Style = 'checkbox';
        S.GUIPanels.Punish = {'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound'}; 
    
        % choice params
        S.GUI.ChoiceWindow_s = 5; % How long after go cue until the mouse must make a choice
        S.GUI.ConfirmLickInterval_s = 0.2; % min interval until choice can be confirmed    
        S.GUI.ChoiceConfirmWindow_s = 5; % time during which correct choice can be confirmed    
        S.GUIPanels.Choice = {'ChoiceWindow_s', 'ConfirmLickInterval_s', 'ChoiceConfirmWindow_s'};

    end
end


function [S] = UpdateMovingSpouts(obj, S, EnableMovingSpouts)
    S.GUI.EnableMovingSpouts = EnableMovingSpouts;
    if (EnableMovingSpouts == 1)
        S.GUI.GoCueVolume_percent = 0;
    else
        S.GUI.GoCueVolume_percent = 100;
    end
end


function [S] = UpdatePassive(obj, S, EnablePassive)
    if (EnablePassive == 1)
        S.GUI.EnablePassive = 1;
        S.GUI.TrainingLevel = 1;
    end
end






    end
end
