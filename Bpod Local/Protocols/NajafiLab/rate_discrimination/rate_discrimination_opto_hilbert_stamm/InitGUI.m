classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings;
    if isempty(fieldnames(S))

        % Optogentic params
        % S.GUI.EnableManualTrialType = 0;
        % S.GUIMeta.EnableManualTrialType.Style = 'checkbox';
        S.GUI.EnManOptoTrialType = 0;
        S.GUIMeta.EnManOptoTrialType.Style = 'checkbox';        
        S.GUI.ManOptoTrialType = 1;
        S.GUIMeta.ManOptoTrialType.Style = 'popupmenu';
        S.GUIMeta.ManOptoTrialType.String = {'Opto Off', 'Opto On'};
        S.GUI.OptoTrialTypeSeq = 3;
        S.GUIMeta.OptoTrialTypeSeq.Style = 'popupmenu';
        S.GUIMeta.OptoTrialTypeSeq.String = {'Random', 'Random First Block', 'Off First Block', 'On First Block'};
        S.GUI.OnFraction = 0.3;
        S.GUI.NumOptoTrialsPerBlock = 50;  
        S.GUIPanels.Opto = {'EnManOptoTrialType', 'ManOptoTrialType', 'OptoTrialTypeSeq', 'OnFraction', 'NumOptoTrialsPerBlock'};

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
        S.GUI.TimeOutPunish = 0;
        S.GUIPanels.ITI_Dist = {'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'ManuallTimeOutPunish', 'TimeOutPunish'};

        % training level params
        S.GUI.NoInit = 0;
        S.GUIMeta.NoInit.Style = 'checkbox';
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 2;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu';
        S.GUIMeta.TrainingLevel.String = {'Passive', 'Naive', 'Mid Trained 1', 'Mid Trained 2', 'Well Trained'};
        S.GUI.NumNaiveWarmup = 15;
        S.GUIPanels.Training = {'NoInit', 'MaxTrials', 'TrainingLevel', 'NumNaiveWarmup'};

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
        S.GUI.PostRewardDelay_s = 2;
        S.GUI.MinISIPerturb_ms = 100;
        S.GUI.PreVisStimDelay_s = 0;
        S.GUI.PreGoCueDelay_s = 0;
        S.GUI.EasyMax = 1;
        S.GUIMeta.EasyMax.Style = 'popupmenu';
        S.GUIMeta.EasyMax.String = {'Default', 'Activated', 'Deactivated'};
        S.GUIPanels.VisStim = {'VisStimEnable', 'RandomOrient', 'PostPertDur', 'ExtraStim', 'PostPertDurExtra', 'GratingDur_s', 'ISIOrig_s', 'PrePertFlashRep', 'PostRewardDelay_s', 'MinISIPerturb_ms', 'PreVisStimDelay_s', 'PreGoCueDelay_s', 'EasyMax'}; 

        % contingency and bias params
        S.GUI.ShortISIFraction = 0.5;
        S.GUI.RepeatedIncorrect = 0;
        S.GUIMeta.RepeatedIncorrect.Style = 'checkbox';
        S.GUI.RepeatedProb = 1.0;
        S.GUI.AdjustValve = 1;
        S.GUIMeta.AdjustValve.Style = 'checkbox';
        S.GUI.NumMonitorTrials = 5;
        S.GUI.BiasIndexThres = 0.5;
        S.GUI.AdjustValvePercent = 0.25;
        S.GUI.AdjustFraction = 0.65;
        S.GUI.FarMoveSpout = 1;
        S.GUIMeta.FarMoveSpout.Style = 'checkbox';
        S.GUI.FarMoveSpoutPos = 12;
        S.GUIPanels.Contingency_Bias = {'ShortISIFraction', 'RepeatedIncorrect', 'RepeatedProb', 'AdjustValve', 'NumMonitorTrials', 'BiasIndexThres', 'AdjustValvePercent', 'AdjustFraction', 'FarMoveSpout', 'FarMoveSpoutPos'};

        % choice params
        S.GUI.ManualChoiceWindow = 0;
        S.GUIMeta.ManualChoiceWindow.Style = 'checkbox';
        S.GUI.ChoiceWindow_s = 5;
        S.GUI.ManuallChangeMindDur = 0;
        S.GUIMeta.ManuallChangeMindDur.Style = 'checkbox';
        S.GUI.ChangeMindDur = 2;
        S.GUI.CenterValveAmount_uL = 0;
        S.GUI.LeftValveAmount_uL = 5;
        S.GUI.RightValveAmount_uL = 5;
        S.GUIPanels.Choice = {'ManualChoiceWindow', 'ChoiceWindow_s', 'ManuallChangeMindDur', 'ChangeMindDur', 'CenterValveAmount_uL', 'LeftValveAmount_uL', 'RightValveAmount_uL'};

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
        S.GUI.OmiAvoidFrame = 3;
        S.GUI.OmiMinInterval = 3;
        S.GUI.SpontSilenceTime = 300;
        S.GUIPanels.Passive = {'EnablePassive', 'SessionMode', 'TrialPerBlock', 'BlockRep', 'ActRandomISI', 'RandomISIMin', 'RandomISIMax', 'RandomISIWid', 'RandomISIStd', 'ActOmi', 'OmiProb', 'OmiAvoidFrame', 'OmiMinInterval', 'SpontSilenceTime'};

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
        S.GUI.RandomOrient = 1;
        S.GUI.PostRewardDelay_s = 0;
        S.GUI.NumNaiveWarmup = 0;
        S.GUI.ActRandomISI = 1;
        S.GUI.AudioStimVolume_percent = 1;
        switch PassiveSessMode
            case 1 % omisison
                S.GUI.SetManualITI = 1;
                S.GUI.ManualITI = S.GUI.ISIOrig_s * 0.75; % trying to approximate 500ms gray, so mouse doesn't notice a change in gray screen duration from trial to trial; but this is approximte!!!
                S.GUI.TrialPerBlock = 8; % xx trials, each including "PrePertFlashRep" reps of grating-gray
                S.GUI.PrePertFlashRep = 150; % reps of grating-gray
                S.GUI.PostPertDur = 0;
                S.GUI.RandomISIWid = 0.4;
                S.GUI.ActOmi = 1;
                S.GUI.OmiProb = 0.20;
                S.GUI.OmiMinInterval = 3; % min gratings before an omission is allowed
                S.GUI.MaxTrials = 4*S.GUI.TrialPerBlock;  % number of blocks x number of trials per block
                S.GUI.SpontSilenceTime = 300; % no-stim interval at the beginning, end, and in between blocks

            case 2 % pre-post 
                S.GUI.SetManualITI = 1;
                S.GUI.ManualITI = 3;
                S.GUI.TrialPerBlock = 60;
                S.GUI.PrePertFlashRep = 10;
                S.GUI.PostPertDur = 6;
                S.GUI.RandomISIMax = 1.15;
                S.GUI.RandomISIWid = 0.25;
                S.GUI.ActOmi = 0;
                S.GUI.MaxTrials = 4*S.GUI.TrialPerBlock;
                S.GUI.SpontSilenceTime = 300;
        end
    end
end



    end
end
