classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings;
    if isempty(fieldnames(S))

        % Servos - spouts
        S.GUI.EnableMovingSpouts = 1;
        S.GUIMeta.EnableMovingSpouts.Style = 'checkbox';
        S.GUI.RightServoInPos = 1175.50;
        S.GUI.LeftServoInPos = 1868.00;
        S.GUI.ServoDeflection = -122.5;
        S.GUIPanels.Servos = {'EnableMovingSpouts', 'RightServoInPos', 'LeftServoInPos', 'ServoDeflection'};

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
        S.GUI.TimeOutPunishMin = 1;
        S.GUI.TimeOutPunishMax = 5;
        S.GUI.TimeOutPunishMean = 3;
        S.GUIPanels.ITI_Dist = {'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'ManuallTimeOutPunish', 'TimeOutPunishMin', 'TimeOutPunishMax', 'TimeOutPunishMean'};

        % training level params
        S.GUI.NoInit = 0;
        S.GUIMeta.NoInit.Style = 'checkbox';
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 6;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu';
        S.GUIMeta.TrainingLevel.String = {'Passive', 'Habituation', 'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumHabituationWarmup = 5;
        S.GUI.NumNaiveWarmup = 15;
        S.GUI.ChangeMindDur = 2;
        S.GUI.ResetWaitDur = 1;
        S.GUIMeta.ResetWaitDur.Style = 'checkbox';
        S.GUI.WaitDurOrig_s = 0.0;
        S.GUI.WaitDurStep_s = 0.01;
        S.GUIPanels.Training = {'NoInit', 'MaxTrials', 'TrainingLevel', 'NumHabituationWarmup', 'NumNaiveWarmup', 'ChangeMindDur', 'ResetWaitDur', 'WaitDurOrig_s', 'WaitDurStep_s'};

        % difficulty params
        S.GUI.PercentEasy = 100;
        S.GUI.PercentMediumEasy = 0;
        S.GUI.PercentMediumHard = 0;
        S.GUI.PercentHard = 0;    
        S.GUIPanels.Difficulty = {'PercentEasy', 'PercentMediumEasy', 'PercentMediumHard', 'PercentHard'};

        % audio stim params
        S.GUI.InitCueVolume_percent = 0.5;
        S.GUI.InitCueDuration_s = 0.05;
        S.GUI.InitWindowTimeout_s = 5;
        S.GUI.InitCueFreq_Hz = 4900;
        S.GUI.GoCueVolume_percent = 100;
        S.GUI.GoCueDuration_s = 0.0;
        S.GUI.GoCueFreq_Hz = 11025;
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1;
        S.GUI.AudioStimFreq_Hz = 11025;
        S.GUIPanels.AudioStim = {'InitCueVolume_percent', 'InitCueDuration_s', 'InitWindowTimeout_s', 'InitCueFreq_Hz', 'GoCueVolume_percent', 'GoCueDuration_s', 'GoCueFreq_Hz', 'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz'};

        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.1;
        S.GUI.ISIOrig_s = 0.5;
        S.GUI.PrePertFlashRep = 2;
        S.GUI.PostPertDur = 3;
        S.GUI.ExtraStim = 1;
        S.GUIMeta.ExtraStim.Style = 'popupmenu';
        S.GUIMeta.ExtraStim.String = {'Default', 'Activated', 'Deactivated'};
        S.GUI.PostPertDurExtra = 3;
        S.GUI.PostRewardDelay_s = 5;
        S.GUI.MinISIPerturb_ms = 100;
        S.GUI.PreVisStimDelay_s = 0;
        S.GUI.PreGoCueDelay_s = 0;
        S.GUI.EasyMax = 1;
        S.GUIMeta.EasyMax.Style = 'popupmenu';
        S.GUIMeta.EasyMax.String = {'Default', 'Activated', 'Deactivated'};
        S.GUIPanels.VisStim = {'VisStimEnable', 'PostPertDur', 'ExtraStim', 'PostPertDurExtra', 'GratingDur_s', 'ISIOrig_s', 'PrePertFlashRep', 'PostRewardDelay_s', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s', 'EasyMax'}; 

        % contingency and bias params
        S.GUI.ShortISIFraction = 0.5;
        S.GUI.RepeatedIncorrect = 0;
        S.GUIMeta.RepeatedIncorrect.Style = 'checkbox';
        S.GUI.RepeatedProb = 1.0;
        S.GUI.AdjustValve = 1;
        S.GUIMeta.AdjustValve.Style = 'checkbox';
        S.GUI.NumMonitorTrials = 2;
        S.GUI.BiasIndexThres = 0.8;
        S.GUI.AdjustValvePercent = 0.12;
        S.GUI.AdjustFraction = 0.65;
        S.GUI.FarMoveSpout = 1;
        S.GUIMeta.FarMoveSpout.Style = 'checkbox';
        S.GUI.FarMoveSpoutPos = 12;
        S.GUIPanels.Contingency_Bias = {'ShortISIFraction', 'RepeatedIncorrect', 'RepeatedProb', 'AdjustValve', 'NumMonitorTrials', 'BiasIndexThres', 'AdjustValvePercent', 'AdjustFraction', 'FarMoveSpout', 'FarMoveSpoutPos'};

        % choice params
        S.GUI.ChoiceWindow_s = 5;
        S.GUI.ConfirmLickInterval_s = 0.2;  
        S.GUI.ChoiceConfirmWindow_s = 5;
        S.GUI.CenterValveAmount_uL = 1;
        S.GUI.LeftValveAmount_uL = 5;
        S.GUI.RightValveAmount_uL = 5;
        S.GUI.IncorrectSoundVolume_percent = 0.15;
        S.GUI.PunishSoundDuration_s = 1;
        S.GUI.IncorrectSound = 1;
        S.GUIMeta.IncorrectSound.Style = 'checkbox';
        S.GUIPanels.Choice = {'CenterValveAmount_uL', 'LeftValveAmount_uL', 'RightValveAmount_uL', 'IncorrectSoundVolume_percent', 'PunishSoundDuration_s', 'IncorrectSound', 'ChoiceWindow_s', 'ConfirmLickInterval_s', 'ChoiceConfirmWindow_s'};

        % passive
        S.GUI.EnablePassive = 0;
        S.GUIMeta.EnablePassive.Style = 'checkbox';
        S.GUI.SessionMode = 1;
        S.GUIMeta.SessionMode.Style = 'popupmenu';
        S.GUIMeta.SessionMode.String = {'Omission', 'PrePost'};
        S.GUI.TrialPerBlock = 1;
        S.GUI.BlockRep = 2;
        S.GUI.ActRandomISI = 0;
        S.GUIMeta.ActRandomISI.Style = 'checkbox';
        S.GUI.RandomISIMin = 0.05;
        S.GUI.RandomISIMax = 0.95;
        S.GUI.RandomISIWid = 0.2;
        S.GUI.RandomISIStd = 0.25;
        S.GUI.ActOmi = 0;
        S.GUIMeta.ActOmi.Style = 'checkbox';
        S.GUI.OmiProb = 0.25;
        S.GUI.OmiAvoidFrame = 5;
        S.GUI.OmiMinInterval = 3;
        S.GUIPanels.Passive = {'EnablePassive', 'SessionMode', 'TrialPerBlock', 'BlockRep', 'ActRandomISI', 'RandomISIMin', 'RandomISIMax', 'RandomISIWid', 'RandomISIStd', 'ActOmi', 'OmiProb', 'OmiAvoidFrame', 'OmiMinInterval'};

    end
end


function [S] = UpdateMovingSpouts(obj, S, EnableMovingSpouts)
    S.GUI.EnableMovingSpouts = EnableMovingSpouts;
    if (EnableMovingSpouts == 1)
        S.GUI.GoCueVolume_percent = 0;
        S.GUI.CenterValveAmount_uL = 0;
    end
end


function [S] = UpdatePassive(obj, S, EnablePassive, PassiveSessMode)
    S.GUI.EnablePassive = EnablePassive;
    S.GUI.SessionMode = PassiveSessMode;
    if (EnablePassive == 1)
        S.GUI.TrainingLevel = 1;
        S.GUI.PostRewardDelay_s = 0;
        S.GUI.NumHabituationWarmup = 0;
        S.GUI.NumNaiveWarmup = 0;
        S.GUI.ActRandomISI = 1;
        switch PassiveSessMode
            case 1
                S.GUI.SetManualITI = 1;
                S.GUI.ManualITI = 5;
                S.GUI.TrialPerBlock = 1;
                S.GUI.PrePertFlashRep = 1200;
                S.GUI.PostPertDur = 0;
                S.GUI.RandomISIWid = 0.4;
                S.GUI.ActOmi = 1;
                S.GUI.OmiProb = 0.20;
                S.GUI.OmiMinInterval = 3;
                S.GUI.MaxTrials = 4;
            case 2
                S.GUI.SetManualITI = 1;
                S.GUI.ManualITI = 3;
                S.GUI.TrialPerBlock = 60;
                S.GUI.PrePertFlashRep = 6;
                S.GUI.PostPertDur = 6;
                S.GUI.RandomISIWid = 0.25;
                S.GUI.ActOmi = 0;
                S.GUI.MaxTrials = 4*S.GUI.TrialPerBlock;
        end
    end
end



    end
end
