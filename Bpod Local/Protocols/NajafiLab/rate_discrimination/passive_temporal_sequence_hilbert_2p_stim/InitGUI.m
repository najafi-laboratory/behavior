classdef InitGUI
    methods


function [S] = SetParams(obj, BpodSystem)
    S = BpodSystem.ProtocolSettings;
    if isempty(fieldnames(S))

        % stim params
        S.GUI.AudioStimEnable = 1;
        S.GUIMeta.AudioStimEnable.Style = 'checkbox';
        S.GUI.AudioStimVolume_percent = 1.0;
        S.GUI.AudioStimFreq_Hz = 11025;
        S.GUI.VisStimEnable = 1;
        S.GUIMeta.VisStimEnable.Style = 'checkbox';
        S.GUI.GratingDur_s = 0.2;
        S.GUIPanels.AudVisStim = {'AudioStimEnable', 'AudioStimVolume_percent', 'AudioStimFreq_Hz', 'VisStimEnable', 'GratingDur_s'};

        % passive general
        S.GUI.MaxImg = 1500;
        S.GUI.OrienBlockNumMin = 10;
        S.GUI.OrienBlockNumMax = 15;
        S.GUI.SpontSilenceTimeSess = 300;
        S.GUIPanels.Passive = {'MaxImg', 'OrienBlockNumMin', 'OrienBlockNumMax', 'SpontSilenceTimeSess'};
        
        % oddball
        S.GUI.OddballMode = 1;
        S.GUIMeta.OddballMode.Style = 'popupmenu';
        S.GUIMeta.OddballMode.String = {'Short', 'Long', 'Random', 'Block'};
        S.GUI.OddAvoidFrameStart = 2;
        S.GUI.OddAvoidFrameEnd = 2;
        S.GUI.OddAvoidFrameBetween = 3;
        S.GUI.ShortNormShortOdd = 1.0;
        S.GUI.ShortNormLongOdd = 2.0;
        S.GUI.LongNormShortOdd = 1.0;
        S.GUI.LongNormLongOdd = 3.0;
        S.GUI.OddProb = 0.35;
        S.GUIPanels.Oddball = {'OddballMode', 'OddAvoidFrameStart', 'OddAvoidFrameEnd', 'OddAvoidFrameBetween', 'ShortNormShortOdd', 'ShortNormLongOdd', 'LongNormShortOdd', 'LongNormLongOdd', 'OddProb'};

        % ISI
        S.GUI.NormalMode = 1;
        S.GUIMeta.NormalMode.Style = 'popupmenu';
        S.GUIMeta.NormalMode.String = {'Short', 'Long', 'Random', 'Block'};
        S.GUI.ShortNormISI = 1.0;
        S.GUI.LongNormISI = 2.0;
        S.GUI.FixJitterMode = 1;
        S.GUIMeta.FixJitterMode.Style = 'popupmenu';
        S.GUIMeta.FixJitterMode.String = {'Fix', 'Jitter', 'Random', 'Block'};
        S.GUI.ShortRandomMin = 0.2;
        S.GUI.ShortRandomMax = 1.8;
        S.GUI.ShortRandomStd = 0.8;
        S.GUI.LongRandomMin = 1.2;
        S.GUI.LongRandomMax = 2.8;
        S.GUI.LongRandomStd = 0.8;
        S.GUIPanels.FixJitterISI = {'NormalMode', 'ShortNormISI', 'LongNormISI', 'FixJitterMode', 'ShortRandomMin', 'ShortRandomMax', 'ShortRandomStd', 'LongRandomMin', 'LongRandomMax', 'LongRandomStd'};

        % Optogentics
        S.GUI.OptoMode = 1;
        S.GUIMeta.OptoMode.Style = 'popupmenu';
        S.GUIMeta.OptoMode.String = {'off', 'on', 'default', 'Random', 'Block'};
        S.GUI.OptoProb = 0.5;
        S.GUI.OptoOnPreStim = 0.2;
        S.GUI.OptoOffPostStim = 0.4;
        S.GUI.OptoIntervalOdd = 0.2;
        S.GUI.LEDOnPulseDur = 0.015;
        S.GUI.LEDOffPulseDur = 0.085;
        S.GUI.OptoAvoidFrameStart = 2;
        S.GUI.OptoAvoidFrameBetween = 1;
        S.GUIPanels.Opto = {'OptoMode', 'OptoProb', 'OptoOnPreStim', 'OptoOffPostStim', 'OptoIntervalOdd', 'LEDOnPulseDur', 'LEDOffPulseDur', 'OptoAvoidFrameStart', 'OptoAvoidFrameBetween'};    
    end
end


function [S] = UpdateOpto(obj, S, OptoMode)
    S.GUI.OptoMode = OptoMode;
end


function [S] = UpdateJitter(obj, S, FixJitterMode)
    S.GUI.FixJitterMode = FixJitterMode;
end


function [S] = UpdateOddballMode(obj, S, OddballMode)
    S.GUI.OddballMode = OddballMode;
end


function [S] = UpdateNormalMode(obj, S, NormalMode)
    S.GUI.NormalMode = NormalMode;
end


    end
end
