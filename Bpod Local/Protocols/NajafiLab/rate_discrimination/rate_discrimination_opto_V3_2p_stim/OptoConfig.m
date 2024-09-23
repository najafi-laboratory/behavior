classdef OptoConfig

    methods


function [OptoType] = GenOptoType(obj, S)
    OptoType = 0 + (rand(1, 1106) < S.GUI.OnFraction);
end


function [AudStim] = OptoUpdateAudStimTrig(obj, S, OptoType, currentTrial)
    switch OptoType(currentTrial)
        case 0
            AudStim = {'HiFi1', ['P', 4]};
        case 1
            if S.GUI.OptoSource == 1 % LED Opto
                AudStim = {'HiFi1', ['P', 4], 'GlobalTimerTrig', '0011'};
            else% 2p stim Opto
                AudStim = {'HiFi1', ['P', 4], 'GlobalTimerTrig', '1000'};
            end
    end
end


function [ResetTimer] = OptoUpdateResetTimer(obj, S, OptoType, currentTrial)
    switch OptoType(currentTrial)
        case 0
            ResetTimer = {};
        case 1
            ResetTimer = {'GlobalTimerCancel', '0011', 'GlobalTimerTrig', '0100'};
    end
end


function [sma] = SetOpto(obj, BpodSystem, S, sma, VisStim, OptoType, currentTrial)
    if (OptoType(currentTrial) == 1) && (S.GUI.OptoSource == 1)
        PMTStartCloseDelay = 0.010;
        PMTCloseTransferDelay = 0.0121;
        PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;          
        PMTMin5VSignalDur = PMTCloseDelay + 0.003;
        PMTStartOpenDelay = 0; % 0.0078
        PMTOpenTransferDelay = 0.0125;
        LEDOnsetDelay = 0.033;
        PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay;
        LEDOnDur = S.GUI.LEDOnPulseDur;
        LEDOffDur = S.GUI.LEDOffPulseDur;
        T = LEDOnDur + LEDOffDur;
        PMT5VDur = max(PMTCloseDelay + LEDOnDur, PMTMin5VSignalDur);
        PMT0VDur =  T - PMT5VDur;
        OptoDuration = VisStim.Data.Pre.Dur + VisStim.Data.Post.Dur + VisStim.Data.Extra.Dur;
        numPMTLEDCycles = floor(OptoDuration/T);
        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', LEDOnDur, 'OnsetDelay', LEDOnsetDelay,...
            'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
            'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);
        sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
            'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
            'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);
        sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', 0.03, 'OnsetDelay', PMT0VDur*2,...
            'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);
    end
end

function [sma] = Set2PStim(obj, BpodSystem, S, sma, VisStim, OptoType, currentTrial)
    if (OptoType(currentTrial) == 1) && (S.GUI.OptoSource == 2)
        VisStimOnsetDelay = 0.033;
        StimOnsetDelay = VisStimOnsetDelay - S.GUI.TimeToFirstLaserPulse;
    
        sma = SetGlobalTimer(sma, 'TimerID', 4, 'Duration', 0.005, 'OnsetDelay', StimOnsetDelay,...
            'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
            'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);
    end
end

    end
end