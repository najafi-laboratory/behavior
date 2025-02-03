classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings;
    if isempty(fieldnames(S))

        S.GUI.ExperimenterInitials = 'Initials';
        S.GUIPanels.SessionInfo = {'ExperimenterInitials'};

        % Optogentic params
        S.GUI.OptoSession = 0;
        S.GUIMeta.OptoSession.Style = 'checkbox';
        S.GUIPanels.Opto = {'OptoSession'};
        S.GUI.LEDOnPulseDur = 0.0078;
        S.GUI.OnFraction = 0.3;
        S.GUI.OptoFreq = 0.1;      
        S.GUIPanels.Opto = {'OptoSession', 'LEDOnPulseDur', 'OnFraction', 'OptoFreq'};
        
        % chemogentic
        S.GUI.ChemoSession = 0;
        S.GUIMeta.ChemoSession.Style = 'checkbox';
        S.GUIPanels.Chemo = {'ChemoSession'};

        % ITI params
        S.GUI.SetManualITI = 0;
        S.GUIMeta.SetManualITI.Style = 'checkbox';
        S.GUI.ManualITI = 0;
        S.GUI.ITIMin = 2;
        S.GUI.ITIMax = 6;
        S.GUI.ITIMean = 4;
        S.GUI.ActTimeOutPunish = 1;
        S.GUIMeta.ActTimeOutPunish.Style = 'checkbox';
        S.GUI.ManuallTimeOutPunish = 0;
        S.GUIMeta.ManuallTimeOutPunish.Style = 'checkbox';
        S.GUI.TimeOutPunish = 2;
        S.GUIPanels.ITI_Dist = {'SetManualITI', 'ManualITI', 'ITIMin', 'ITIMax', 'ITIMean', 'ActTimeOutPunish', 'ManuallTimeOutPunish', 'TimeOutPunish'};

        % training level params
        S.GUI.NoInit = 0;
        S.GUIMeta.NoInit.Style = 'checkbox';
        S.GUI.MaxTrials = 1000;
        S.GUI.TrainingLevel = 2;
        S.GUIMeta.TrainingLevel.Style = 'popupmenu';
        S.GUIMeta.TrainingLevel.String = {'Naive', 'Early', 'Mid 1', 'Mid 2', 'Well'};
        S.GUI.NumNaiveWarmup = 15;
        S.GUI.MaxSameSide = 3;
        S.GUIPanels.Training = {'NoInit', 'MaxTrials', 'TrainingLevel', 'NumNaiveWarmup', 'MaxSameSide'};
        
        % Servos - spouts
        S.GUI.EnableMovingSpouts = 1;
        S.GUIMeta.EnableMovingSpouts.Style = 'checkbox';
        switch BpodSystem.Data.RigName
            case 'ImagingRig'
                S.GUI.RightServoInPos = 1160;
                S.GUI.LeftServoInPos = 1690;
            case '2AFCRig2'
                S.GUI.RightServoInPos = 1138.00;  % close
                S.GUI.LeftServoInPos = 1902.00;   % close
                % S.GUI.RightServoInPos = 1148.00;    % med
                % S.GUI.LeftServoInPos = 1892.00;     % med           
                % S.GUI.RightServoInPos = 1136.00;
                % S.GUI.LeftServoInPos = 1904.00;
            case '2AFCRig1'
                S.GUI.RightServoInPos = 1076.00;    % close
                S.GUI.LeftServoInPos = 1784.00;     % close
                % S.GUI.RightServoInPos = 1080.00;    % med
                % S.GUI.LeftServoInPos = 1770.00;     % med                
        end

        S.GUI.ServoDeflection = -100;
        S.GUI.ServoVelocity = 1;
        S.GUI.AntiBiasServoAdjustAct = 0;
        S.GUIMeta.AntiBiasServoAdjustAct.Style = 'checkbox';        
        S.GUI.ServoBiasIncrement = 4;
        S.GUI.ServoIncrementMax = 25;
        S.GUI.ResetServoAdjust = 0;
        S.GUIMeta.ResetServoAdjust.Style = 'checkbox';

        S.GUIPanels.Servos = {'EnableMovingSpouts', 'RightServoInPos', 'LeftServoInPos', 'ServoDeflection', 'ServoVelocity', 'AntiBiasServoAdjustAct', 'ServoBiasIncrement', 'ServoIncrementMax', 'ResetServoAdjust'};

        % audio stim params
        S.GUI.InitCueVolume_percent = 0.35;
        S.GUI.InitCueDuration_s = 0.05;
        S.GUI.InitWindowTimeout_s = 5;
        S.GUI.InitCueFreq_Hz = 4900;
        S.GUI.GoCueVolume_percent = 0;
        S.GUI.GoCueDuration_s = 0.0;
        S.GUI.GoCueFreq_Hz = 14700;
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 0.1;
        S.GUI.AudioStimFreq_Hz = 11025;
        % S.GUI.AudioStimFreq_Hz = 16000; % near center of mouse hearing sensitivity
        S.GUI.NoiseVolume_percent = 0;
        S.GUI.NoiseDuration_s = 0;
        S.GUI.ActNoise = 1;
        S.GUIMeta.ActNoise.Style = 'checkbox';
        S.GUIPanels.AudioStim = {'InitCueVolume_percent', 'InitCueDuration_s', 'InitWindowTimeout_s', 'InitCueFreq_Hz', 'GoCueVolume_percent', 'GoCueDuration_s', 'GoCueFreq_Hz', 'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz', 'NoiseVolume_percent', 'NoiseDuration_s', 'ActNoise'};

        % difficulty params
        S.GUI.PercentEasy = 100;
        S.GUI.PercentMediumEasy = 0;
        S.GUI.PercentMediumHard = 0;
        S.GUI.PercentHard = 0;
        S.GUIPanels.Difficulty = {'PercentEasy', 'PercentMediumEasy', 'PercentMediumHard', 'PercentHard'};


        % vis stim params
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.RandomOrient = 0;
        S.GUIMeta.RandomOrient.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.2;
        % default ISIs
        S.GUI.OddballISIShortMean_s = 0.200;
        S.GUI.OddballISIShortMin_s = 0.000;
        S.GUI.OddballISIShortMax_s = 0.700; 
        S.GUI.OddballISILongMean_s = 1.200;
        S.GUI.OddballISILongMin_s = 0.700;
        S.GUI.OddballISILongMax_s = 1.400;    
        % S.GUI.ISIShortMin_s = 0.050;
        S.GUI.ISIShortMean_s = 0.200;
        S.GUI.ISIShortMin_s = 0.000;
        S.GUI.ISIShortMax_s = 0.700;        
        % S.GUI.ISIShortMax_s = 0.750;
        % S.GUI.ISILongMin_s = 0.750;
        % S.GUI.ISILongMean_s = 1.100;
        % S.GUI.ISILongMax_s = 1.450;                
        S.GUI.ISILongMean_s = 1.200;
        S.GUI.ISILongMin_s = 0.700;
        S.GUI.ISILongMax_s = 1.400;
        S.GUI.ISIOrig_s = 0.700;        
        % S.GUI.ISIOrig_s = 2;
        % S.GUI.PrePertFlashRep = 2;
        S.GUI.PrePertFlashRep = 0;
        S.GUI.PostPertDur = 3;
        S.GUI.ExtraStim = 1;
        S.GUIMeta.ExtraStim.Style = 'popupmenu';
        S.GUIMeta.ExtraStim.String = {'Default', 'Manual', 'Zero'};
        S.GUI.PostPertDurExtra = 3;
        S.GUI.PreRewardDelay = 0;
        S.GUI.PostRewardDelay = 3;
        S.GUI.PostPunishDelay = 0;
        S.GUI.PostOutcomeDelay = 0;
        S.GUI.MinISIPerturb_ms = 100;
        S.GUI.PreVisStimDelayMin_s = 0.200;
        S.GUI.PreVisStimDelayMean_s = 0.500;
        S.GUI.PreVisStimDelayMax_s = 0.800;        
        S.GUI.PostVisStimDelayMin_s = 0.050;
        S.GUI.PostVisStimDelayMean_s = 0.200;
        S.GUI.PostVisStimDelayMax_s = 0.200;        
        S.GUI.PreGoCueDelay_s = 0;
        S.GUI.EasyMax = 1;
        S.GUIMeta.EasyMax.Style = 'popupmenu';
        S.GUIMeta.EasyMax.String = {'Default', 'Activated', 'Deactivated'};
        S.GUIPanels.VisStim = {'VisStimEnable', 'RandomOrient', 'PostPertDur', 'ExtraStim', 'PostPertDurExtra', 'GratingDur_s', 'OddballISIShortMean_s', 'OddballISIShortMin_s', 'OddballISIShortMax_s', 'OddballISILongMean_s', 'OddballISILongMin_s', 'OddballISILongMax_s', 'ISIShortMin_s', 'ISIShortMean_s', 'ISIShortMax_s', 'ISILongMin_s', 'ISILongMean_s', 'ISILongMax_s', 'ISIOrig_s', 'PrePertFlashRep', 'PreRewardDelay', 'PostRewardDelay', 'PostPunishDelay', 'PostOutcomeDelay', 'MinISIPerturb_ms', 'PreVisStimDelayMin_s', 'PreVisStimDelayMean_s', 'PreVisStimDelayMax_s', 'PostVisStimDelayMin_s', 'PostVisStimDelayMean_s', 'PostVisStimDelayMax_s', 'PreGoCueDelay_s', 'EasyMax'}; 

        % contingency and bias params
        S.GUI.ShortISIFraction = 0.5;
        S.GUI.RepeatedIncorrect = 0;
        S.GUIMeta.RepeatedIncorrect.Style = 'checkbox';
        S.GUI.RepeatedProb = 1.0;
        S.GUI.AdjustValve = 0;
        S.GUIMeta.AdjustValve.Style = 'checkbox';
        S.GUI.NumMonitorTrials = 7;
        S.GUI.BiasIndexThres = 0.5;
        S.GUI.AdjustValvePercent = 0.25;
        % S.GUI.AdjustValvePercent = 1;
        S.GUI.AdjustFractionAct = 0;
        S.GUIMeta.AdjustFractionAct.Style = 'checkbox';  
        % fraction of unfavored side
        S.GUI.AdjustFraction = 0.7;
        S.GUI.ManualSideAct = 0;
        S.GUIMeta.ManualSideAct.Style = 'checkbox';
        S.GUI.ManualSide = 1;
        S.GUIMeta.ManualSide.Style = 'popupmenu';
        S.GUIMeta.ManualSide.String = {'Left', 'Right'};
        S.GUI.AntiBiasProbeAct = 1;
        S.GUIMeta.AntiBiasProbeAct.Style = 'checkbox';     
        S.GUI.AutoSingleSpout = 1;
        S.GUIMeta.AutoSingleSpout.Style = 'checkbox';        
        S.GUI.ProbeWaterDistribution = 2;
        S.GUIMeta.ProbeWaterDistribution.Style = 'popupmenu';
        S.GUIMeta.ProbeWaterDistribution.String = {'Water Off', 'Water On', 'Random'};        
        S.GUI.ManualSingleSpoutAct = 0;
        S.GUIMeta.ManualSingleSpoutAct.Style = 'checkbox';          
        S.GUIPanels.Contingency_Bias = {'ShortISIFraction', 'RepeatedIncorrect', 'RepeatedProb', 'AdjustValve', 'NumMonitorTrials', 'BiasIndexThres', 'AdjustValvePercent', 'AdjustFractionAct', 'AdjustFraction', 'ManualSideAct', 'ManualSide', 'AntiBiasProbeAct', 'AutoSingleSpout', 'ProbeWaterDistribution', 'ManualSingleSpoutAct'};

        S.GUI.ProbeTrialsAct = 0;
        S.GUIMeta.ProbeTrialsAct.Style = 'checkbox';
        S.GUI.ProbeTrialsFraction = 0.10;
        S.GUIPanels.ProbeTrials = {'ProbeTrialsAct', 'ProbeTrialsFraction'};

        % choice params
        S.GUI.ChoiseWindowStartDelay = 0;
        S.GUI.ManualChoiceWindow = 0;
        S.GUIMeta.ManualChoiceWindow.Style = 'checkbox';
        S.GUI.ChoiceWindow_s = 5;
        S.GUI.ManuallChangeMindDur = 0;
        S.GUIMeta.ManuallChangeMindDur.Style = 'checkbox';
        S.GUI.ChangeMindDur = 2;
        S.GUI.CenterValveAmount_uL = 0;
        S.GUI.LeftValveAmount_uL = 4;
        S.GUI.RightValveAmount_uL = 4;
        S.GUI.OutcomeFeedbackDelay = 0;
        S.GUIPanels.Choice = {'ChoiseWindowStartDelay', 'ManualChoiceWindow', 'ChoiceWindow_s', 'ManuallChangeMindDur', 'ChangeMindDur', 'CenterValveAmount_uL', 'LeftValveAmount_uL', 'RightValveAmount_uL', 'OutcomeFeedbackDelay'};

        % jitter
        S.GUI.ReactionTask = 0;
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


% function [S] = UpdateOpto(obj, S, EnableOpto)
%     if (EnableOpto == 0)
%         S.GUI.OnFraction = 0;
%     end
% end


    end
end
