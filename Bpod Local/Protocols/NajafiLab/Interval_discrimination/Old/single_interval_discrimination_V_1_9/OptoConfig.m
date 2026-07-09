classdef OptoConfig

methods


function [OptoType] = GenOptoType(obj, S)
    OptoType = zeros(1, S.GUI.MaxTrials);
    if S.GUI.OptoSession
        OptoType = 0 + (rand(1, S.GUI.MaxTrials) < S.GUI.OnFraction);
    end
end


function [AudStim] = OptoUpdateAudStimTrig(obj, OptoType, currentTrial)
    switch OptoType(currentTrial)
        case 0
            AudStim = {'HiFi1', ['P', 4]};
        case 1
            AudStim = {'HiFi1', ['P', 4], 'GlobalTimerTrig', '11'};
    end
end


function [sma] = SetOpto(obj, BpodSystem, S, sma, OptoDuration, OptoType, currentTrial)
    if (OptoType(currentTrial) == 1)
        if ~S.GUI.EnDoricOpto
            PMTStartCloseDelay = 0.010;
            PMTCloseTransferDelay = 0.0121;
            PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;          
            PMTMin5VSignalDur = PMTCloseDelay + 0.003;
            PMTStartOpenDelay = 0.0078; 
            PMTOpenTransferDelay = 0.0125;
            LEDOnsetDelay = 0; 
            PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay;
            LEDOffDur = S.GUI.OptoFreq - S.GUI.LEDOnPulseDur;        
            PMT5VDur = PMTCloseDelay;
            if S.GUI.LEDOnPulseDur > PMTStartOpenDelay
                PMT5VDur = PMT5VDur + (S.GUI.LEDOnPulseDur - PMTStartOpenDelay);
            end
            PMT5VDur = max(PMT5VDur, PMTMin5VSignalDur);
            PMT0VDur =  S.GUI.OptoFreq - PMT5VDur;
            numPMTLEDCycles = floor(OptoDuration / S.GUI.OptoFreq);
            sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', S.GUI.LEDOnPulseDur, 'OnsetDelay', LEDOnsetDelay,...
                'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
                'GlobalTimerEvents', 0, 'OffsetValue', 0);
            sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
                'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
                'GlobalTimerEvents', 0, 'OffsetValue', 0);
        else
            sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', 0.01, 'OnsetDelay', 0,...
                'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                'GlobalTimerEvents', 0, 'OffsetValue', 0);            
        end
    end
end



    end
end