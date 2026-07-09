classdef OptoConfig

methods


function [OptoType] = GenOptoType(obj, S)
    OptoType = 0 + (rand(1, 1106) < S.GUI.OnFraction);
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
        % initial gray frame vis stim offset, statistical delay of
        % 2 frames at 60fps
        VisStimShift = 0;
        switch BpodSystem.Data.RigName
            case 'ImagingRig'
                % % measured gray f1, f2 duration
                % VisStimShift = 0.032292;
                VisStimShift = 0.0329645; % f1 + f2 - imaging rig              
            case 'JoystickRig'
                % VisStimShift = 0.031698; % f1 + f2 - joystick rig
                f1 = 0.0153655; % gray f1 dur
                f2 = 0.0176069; % gray f2 dur
                VisStimShift = f1 + f2; % f1 + f2 - joystick rig
            case 'Rig2'
                % % measured gray f1, f2 duration
                VisStimShift = 0.0325; % f1 + f2 - imaging rig
        end

        PMTStartCloseDelay = 0.010;
        PMTCloseTransferDelay = 0.0121;
        PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;          
        PMTMin5VSignalDur = PMTCloseDelay + 0.003;

        PMTStartOpenDelay = 0.0078; 
        PMTOpenTransferDelay = 0.0125;

        % LEDOnsetDelay = 0; 
        LEDOnsetDelay = VisStimShift; 

        PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay;

        % LEDOffDur = S.GUI.OptoFreq - S.GUI.LEDOnPulseDur;
        % LED waveform is the on/off times for each cycle, looped as a
        % continuous signal in the global timer triggers
        LEDOnDur = S.GUI.LEDOnPulseDur_ms/1000;
        LEDOffDur = S.GUI.LEDOffPulseDur_ms/1000;
        % LED pulse period is on time + off time
        T = LEDOnDur + LEDOffDur;
             
        PMT5VDur = PMTCloseDelay;
        % if S.GUI.LEDOnPulseDur > PMTStartOpenDelay
        %     PMT5VDur = PMT5VDur + (S.GUI.LEDOnPulseDur - PMTStartOpenDelay);
        % end
        % if the LED is on for longer than the shutter StartOpenDelay,
        % then increase shutter 5V duration by the difference (LEDOnDur - PMTStartOpenDelay)
        if LEDOnDur > PMTStartOpenDelay
            PMT5VDur = PMT5VDur + (LEDOnDur - PMTStartOpenDelay);
        end

        % if shutter duration is less than the minimum dur for the
        % shutter to re-open, set it to minimum shutter pulse dur
        PMT5VDur = max(PMT5VDur, PMTMin5VSignalDur);
        % duration of LED pulse 0V is cycle period minus 5V dur
        PMT0VDur =  T - PMT5VDur;
        % PMT0VDur =  S.GUI.OptoFreq - PMT5VDur;        
        % integer number of pmt/led cycles within pre + post grating onset
        numPMTLEDCycles = floor(min(S.GUI.MaxOptoDur_s, OptoDuration)/T);        
        % numPMTLEDCycles = floor(OptoDuration / S.GUI.OptoFreq);
        sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', LEDOnDur, 'OnsetDelay', LEDOnsetDelay,...
            'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
            'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);
        sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
            'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
            'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
            'GlobalTimerEvents', 0, 'OffsetValue', 0);
    end
end


    end
end