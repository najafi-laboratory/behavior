classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings;
    if isempty(fieldnames(S))

        % Optogentic params
        S.GUI.SessionType = 1;
        S.GUIMeta.SessionType.Style = 'popupmenu';
        S.GUIMeta.SessionType.String = {'Opto', 'Control'};        
        S.GUI.PulseType = 2;
        S.GUIMeta.PulseType.Style = 'popupmenu';
        S.GUIMeta.PulseType.String = {'On', 'Square', 'Sinusoidal'};
        S.GUI.MaxOptoDur = 2.5;
        S.GUI.PulseFreq_Hz = 50;
        S.GUI.PulseOnDur_ms = 5;   
        S.GUI.LEDOnPulseDur = 0.0078;
        S.GUI.OptoFreq = 0.100;
        S.GUI.OptoTrialTypeSeq = 1;
        S.GUIMeta.OptoTrialTypeSeq.Style = 'popupmenu';
        S.GUIMeta.OptoTrialTypeSeq.String = {'Random', 'Random First Block', 'Off First Block', 'On First Block'};
        S.GUI.OnFraction = 1; % S.GUI.OnFraction = 0.4;
        S.GUI.NumOptoTrialsPerBlock = 50;          
        S.GUIPanels.Opto = {'SessionType', 'PulseType', 'MaxOptoDur', 'PulseFreq_Hz', 'PulseOnDur_ms', 'LEDOnPulseDur', 'OptoFreq', 'OptoTrialTypeSeq', 'OnFraction', 'NumOptoTrialsPerBlock'};
        
        % chemogentic
        S.GUI.ChemoSession = 0;
        S.GUIMeta.ChemoSession.Style = 'checkbox';
        S.GUIPanels.Chemo = {'ChemoSession'};

        % ITI params
        S.GUI.SetManualITI = 0;
        S.GUIMeta.SetManualITI.Style = 'checkbox';
        S.GUI.ManualITI = 0;
        S.GUI.ITIMin = 1;
        S.GUI.ITIMax = 5;
        S.GUI.ITIMean = 3;
        S.GUI.ActTimeOutPunish = 1;
        S.GUIMeta.ActTimeOutPunish.Style = 'checkbox';
        S.GUI.ManuallTimeOutPunish = 1;
        S.GUIMeta.ManuallTimeOutPunish.Style = 'checkbox';
        S.GUI.TimeOutPunish = 1;
        S.GUIPanels.ITI_Dist = {'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'ManuallTimeOutPunish', 'TimeOutPunish'};

        % training level params
        S.GUI.NoInit = 0;
        S.GUIMeta.NoInit.Style = 'checkbox';
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 4;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu';
        S.GUIMeta.TrainingLevel.String = {'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumNaiveWarmup = 15;
        S.GUIPanels.Training = {'NoInit', 'MaxTrials', 'TrainingLevel', 'NumNaiveWarmup'};
        
        % Servos - spouts
        S.GUI.EnableMovingSpouts = 1;
        S.GUIMeta.EnableMovingSpouts.Style = 'checkbox';
        S.GUI.RightServoInPos = 1130.50;
        S.GUI.LeftServoInPos = 1711.00;
        S.GUI.ServoDeflection = -122.5;
        S.GUIPanels.Servos = {'EnableMovingSpouts', 'RightServoInPos', 'LeftServoInPos', 'ServoDeflection'};

        % difficulty params
        S.GUI.PercentEasy = 100;
        S.GUI.PercentMediumEasy = 0;
        S.GUI.PercentMediumHard = 0;
        S.GUI.PercentHard = 0;
        S.GUIPanels.Difficulty = {'PercentEasy', 'PercentMediumEasy', 'PercentMediumHard', 'PercentHard'};

        % audio stim params
        S.GUI.InitCueVolume_percent = 0.35;
        S.GUI.InitCueDuration_s = 0.05;
        S.GUI.InitWindowTimeout_s = 5;
        S.GUI.InitCueFreq_Hz = 4900;
        S.GUI.GoCueVolume_percent = 100;
        S.GUI.GoCueDuration_s = 0.0;
        S.GUI.GoCueFreq_Hz = 14700;
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 0.9;
        S.GUI.AudioStimFreq_Hz = 11025;
        S.GUI.NoiseVolume_percent = 0.065;
        S.GUI.NoiseDuration_s = 1;
        S.GUI.ActNoise = 1;
        S.GUIMeta.ActNoise.Style = 'checkbox';
        S.GUIPanels.AudioStim = {'InitCueVolume_percent', 'InitCueDuration_s', 'InitWindowTimeout_s', 'InitCueFreq_Hz', 'GoCueVolume_percent', 'GoCueDuration_s', 'GoCueFreq_Hz', 'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz', 'NoiseVolume_percent', 'NoiseDuration_s', 'ActNoise'};

        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.RandomOrient = 0;
        S.GUIMeta.RandomOrient.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.1;
        S.GUI.ISIOrig_s = 0.5;
        S.GUI.PrePertFlashRep = 2;
        S.GUI.PostPertDur = 3;
        S.GUI.ExtraStim = 1;
        S.GUIMeta.ExtraStim.Style = 'popupmenu';
        S.GUIMeta.ExtraStim.String = {'Default', 'Manual', 'Zero'};
        S.GUI.PostPertDurExtra = 3;
        S.GUI.PostOutcomeDelay = 2;
        S.GUI.MinISIPerturb_ms = 100;
        S.GUI.PreVisStimDelay_s = 0;
        S.GUI.PreGoCueDelay_s = 0;
        S.GUI.EasyMax = 1;
        S.GUIMeta.EasyMax.Style = 'popupmenu';
        S.GUIMeta.EasyMax.String = {'Default', 'Activated', 'Deactivated'};
        S.GUIPanels.VisStim = {'VisStimEnable', 'RandomOrient', 'PostPertDur', 'ExtraStim', 'PostPertDurExtra', 'GratingDur_s', 'ISIOrig_s', 'PrePertFlashRep', 'PostOutcomeDelay', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s', 'EasyMax'}; 

        % contingency and bias params
        S.GUI.ShortISIFraction = 0.5;
        S.GUI.RepeatedIncorrect = 0;
        S.GUIMeta.RepeatedIncorrect.Style = 'checkbox';
        S.GUI.RepeatedProb = 1.0;
        S.GUI.AdjustValve = 1;
        S.GUIMeta.AdjustValve.Style = 'checkbox';
        S.GUI.NumMonitorTrials = 3;
        S.GUI.BiasIndexThres = 0.5;
        S.GUI.AdjustValvePercent = 0.25;
        S.GUI.AdjustFraction = 0.9;
        S.GUI.FarMoveSpout = 1;
        S.GUIMeta.FarMoveSpout.Style = 'checkbox';
        S.GUI.FarMoveSpoutPos = 12;
        S.GUIPanels.Contingency_Bias = {'ShortISIFraction', 'RepeatedIncorrect', 'RepeatedProb', 'AdjustValve', 'NumMonitorTrials', 'BiasIndexThres', 'AdjustValvePercent', 'AdjustFraction', 'FarMoveSpout', 'FarMoveSpoutPos'};

        % choice params
        S.GUI.ChoiseWindowStartDelay = 0.1;
        S.GUI.ManualChoiceWindow = 0;
        S.GUIMeta.ManualChoiceWindow.Style = 'checkbox';
        S.GUI.ChoiceWindow_s = 5;
        S.GUI.ManuallChangeMindDur = 0;
        S.GUIMeta.ManuallChangeMindDur.Style = 'checkbox';
        S.GUI.ChangeMindDur = 2;
        S.GUI.CenterValveAmount_uL = 0;
        S.GUI.LeftValveAmount_uL = 5;
        S.GUI.RightValveAmount_uL = 5;
        S.GUI.OutcomeFeedbackDelay = 0;
        S.GUIPanels.Choice = {'ChoiseWindowStartDelay', 'ManualChoiceWindow', 'ChoiceWindow_s', 'ManuallChangeMindDur', 'ChangeMindDur', 'CenterValveAmount_uL', 'LeftValveAmount_uL', 'RightValveAmount_uL', 'OutcomeFeedbackDelay'};

        % jitter
        S.GUI.ReactionTask = 1;
        S.GUIMeta.ReactionTask.Style = 'checkbox';
        S.GUI.ActRandomISI = 0;
        S.GUIMeta.ActRandomISI.Style = 'checkbox';
        S.GUI.RandomISIMin = 0.05;
        S.GUI.RandomISIMax = 0.95;
        S.GUI.RandomISIWid = 0.15;
        S.GUI.RandomISIStd = 0.10;
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
