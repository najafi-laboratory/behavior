classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings;
    if isempty(fieldnames(S))

        % Optogentic params
        S.GUI.SessionType = 2;
        S.GUIMeta.SessionType.Style = 'popupmenu';
        S.GUIMeta.SessionType.String = {'Opto', 'Control'};        
        S.GUI.PulseType = 2;
        S.GUIMeta.PulseType.Style = 'popupmenu';
        S.GUIMeta.PulseType.String = {'On', 'Square', 'Sinusoidal'};
        S.GUI.PulseFreq_Hz = 50;
        S.GUI.PulseOnDur_ms = 5;        
        S.GUI.OptoTrialTypeSeq = 1;
        S.GUIMeta.OptoTrialTypeSeq.Style = 'popupmenu';
        S.GUIMeta.OptoTrialTypeSeq.String = {'Random', 'Random First Block', 'Off First Block', 'On First Block'};
        S.GUI.OnFraction = 0.4;
        S.GUI.NumOptoTrialsPerBlock = 50;          
        S.GUIPanels.Opto = {'SessionType', 'PulseType', 'PulseFreq_Hz', 'PulseOnDur_ms','OptoTrialTypeSeq', 'OnFraction', 'NumOptoTrialsPerBlock'};

        % ITI params
        S.GUI.SetManualITI = 0;
        S.GUIMeta.SetManualITI.Style = 'checkbox';
        S.GUI.ManualITI = 0;
        S.GUI.ITIMin = 1;
        S.GUI.ITIMax = 5;
        S.GUI.ITIMean = 3;
        S.GUI.ActTimeOutPunish = 1;
        S.GUIMeta.ActTimeOutPunish.Style = 'checkbox';
        S.GUI.ManuallTimeOutPunish = 0;
        S.GUIMeta.ManuallTimeOutPunish.Style = 'checkbox';
        S.GUI.TimeOutPunish = 0;
        S.GUIPanels.ITI_Dist = {'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'ManuallTimeOutPunish', 'TimeOutPunish'};

        % training level params
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 1;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu';
        S.GUIMeta.TrainingLevel.String = {'Naive', 'Mid Trained', 'Well Trained'};
        S.GUI.NumNaiveWarmup = 15;
        S.GUIPanels.Training = {'MaxTrials', 'TrainingLevel', 'NumNaiveWarmup'};
        
        % Servos - spouts
        S.GUI.EnableMovingSpouts = 1;
        S.GUIMeta.EnableMovingSpouts.Style = 'checkbox';
        S.GUI.RightServoInPos = 1175.50;
        S.GUI.LeftServoInPos = 1868.00;
        S.GUI.ServoDeflection = -122.5;
        S.GUIPanels.Servos = {'EnableMovingSpouts', 'RightServoInPos', 'LeftServoInPos', 'ServoDeflection'};

        % audio stim params
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 0.9;
        S.GUI.AudioStimFreq_Hz = 11025;
        S.GUI.NoiseVolume_percent = 0.065;
        S.GUI.NoiseDuration_s = 1;
        S.GUI.ActNoise = 1;
        S.GUIMeta.ActNoise.Style = 'checkbox';
        S.GUIPanels.AudioStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz', 'NoiseVolume_percent', 'NoiseDuration_s', 'ActNoise'};

        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.RandomOrient = 0;
        S.GUIMeta.RandomOrient.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.1;
        S.GUI.ISIOrig_s = 0.5;
        S.GUI.VisStimNum = 3;
        S.GUI.PostRewardDelay_s = 2;
        S.GUI.MinISIPerturb_ms = 100;
        S.GUI.PreVisStimDelay_s = 0;
        S.GUI.PreGoCueDelay_s = 0;
        S.GUIMeta.EasyMax.Style = 'popupmenu';
        S.GUIMeta.EasyMax.String = {'Default', 'Activated', 'Deactivated'};
        S.GUIPanels.VisStim = {'VisStimEnable', 'RandomOrient', 'GratingDur_s', 'ISIOrig_s', 'VisStimNum', 'PostRewardDelay_s', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s'}; 

        % choice params
        S.GUI.ManualChoiceWindow = 0;
        S.GUIMeta.ManualChoiceWindow.Style = 'checkbox';
        S.GUI.ChoiceWindow_s = 5;
        S.GUI.CenterValveAmount_uL = 0;
        S.GUI.OutcomeFeedbackDelay = 0;
        S.GUIPanels.Choice = {'ManualChoiceWindow', 'ChoiceWindow_s', 'CenterValveAmount_uL', 'OutcomeFeedbackDelay'};

        % jitter
        S.GUI.ReactionTask = 1;
        S.GUIMeta.ReactionTask.Style = 'checkbox';
        S.GUI.ActRandomISI = 0;
        S.GUIMeta.ActRandomISI.Style = 'checkbox';
        S.GUI.RandomISIMin = 0.05;
        S.GUI.RandomISIMax = 0.95;
        S.GUI.RandomISIWid = 0.2;
        S.GUI.RandomISIStd = 0.25;
        S.GUIPanels.jitter = {'ReactionTask', 'ActRandomISI', 'RandomISIMin', 'RandomISIMax', 'RandomISIWid', 'RandomISIStd'};

    end
end


function [S] = UpdateMovingSpouts(obj, S, EnableMovingSpouts)
    S.GUI.EnableMovingSpouts = EnableMovingSpouts;
    if (EnableMovingSpouts == 1)
        S.GUI.GoCueVolume_percent = 0;
        S.GUI.CenterValveAmount_uL = 0;
    end
end


    end
end
