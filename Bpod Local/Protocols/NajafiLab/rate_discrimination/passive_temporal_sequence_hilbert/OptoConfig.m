classdef OptoConfig
    methods

% get pre and post isi for the current stim
function [PreISI, OddISI, PostISI] = GetPrePostISIOdd( ...
        obj, ISIseq, currentImg)
    paddedISIseq = [ISIseq(1), ISIseq, ISIseq(end)];
    PreISI  = paddedISIseq(currentImg);
    OddISI  = paddedISIseq(currentImg + 1);
    PostISI = paddedISIseq(currentImg + 2);
    PreISI  = PreISI / 2;
    PostISI = PostISI / 2;
end

% get pre and post isi for the current stim
function [PreISI, PostISI] = GetPrePostISI( ...
        obj, ISIseq, currentImg)
    paddedISIseq = [ISIseq(1), ISIseq, ISIseq(end)];
    PreISI  = paddedISIseq(currentImg);
    PostISI = paddedISIseq(currentImg + 1);
    PreISI  = PreISI / 2;
    PostISI = PostISI / 2;
end

function [OptoSeq] = GetOptoSeq(obj, S, TrialTypes, OptoTypes, ISIseq)
    OptoSeq = [];
    currentImg = 1;
    while currentImg <= S.GUI.MaxImg
        if TrialTypes(currentImg) == 1
            [PreISI, OddISI, ~] = GetPrePostISIOdd(obj, ISIseq, currentImg);
            % oddball
            if OptoTypes(currentImg) == 1
                OptoSetup.Type = 1;
                OptoSetup.TimeOn = PreISI + S.GUI.GratingDur_s + S.GUI.OptoIntervalOdd;
                OptoSetup.TimeDur = OddISI - 2*S.GUI.OptoIntervalOdd;
                OptoSeq = [OptoSeq OptoSetup];
            % post oddball
            elseif OptoTypes(currentImg+1) == 2
                OptoSetup.Type = 2;
                OptoSetup.TimeOn = PreISI + S.GUI.GratingDur_s + OddISI - S.GUI.OptoOnPreStim;
                OptoSetup.TimeDur = S.GUI.OptoOnPreStim + S.GUI.GratingDur_s + S.GUI.OptoOffPostStim;
                OptoSeq = [OptoSeq OptoSetup];
            % none
            else
                OptoSetup.Type = 0;
                OptoSetup.TimeOn = 0;
                OptoSetup.TimeDur = 0;
                OptoSeq = [OptoSeq OptoSetup];
            end
            currentImg = currentImg + 2;
        else
            % normal
            if OptoTypes(currentImg) == 3
                OptoSetup.Type = 3;
                [PreISI, ~] = GetPrePostISI(obj, ISIseq, currentImg);
                OptoSetup.TimeOn = PreISI - S.GUI.OptoOnPreStim;
                OptoSetup.TimeDur = S.GUI.OptoOnPreStim + S.GUI.GratingDur_s + S.GUI.OptoOffPostStim;
                OptoSeq = [OptoSeq OptoSetup];
            % none
            else
                OptoSetup.Type = 0;
                OptoSetup.TimeOn = 0;
                OptoSetup.TimeDur = 0;
                OptoSeq = [OptoSeq OptoSetup];
            end
            currentImg = currentImg + 1;
        end
    end
end

function [sma] = SetOpto(obj, S, sma, OptoSeq, currentTrial)
    OptoSetup = OptoSeq(currentTrial);
    % pmt shutter delay timings are from Bruker specs, the actual
    % timings are a bit different, and vary between green/red PMTs, but these work for pulsing opto
    % shutter close delays
    PMTStartCloseDelay = 0.010;     % ~10ms for shutter to start closing after setting shutter signal to 5V
    PMTCloseTransferDelay = 0.0121; % ~12.1ms
    PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;          
    PMTMin5VSignalDur = PMTCloseDelay + 0.003; % ~25.1ms, minimum measured duration of 5V shutter signal for shutter to re-open 
    % shutter open delays
    PMTStartOpenDelay = 0.0078;     % ~7.8ms for shutter to start opening after setting shutter signal to 0V
    PMTOpenTransferDelay = 0.0125;  % ~12.5ms for shutter to open after the 7.8ms open transfer delay
    % opto cycle timing is defined by the onset of the LED
    LEDOnsetDelay = OptoSetup.TimeOn; 
    PMTOnsetDelay = LEDOnsetDelay - PMTCloseDelay;
    % 10Hz pulsed shutter/opto, 7.8ms LED
    LEDOnPulseDur = S.GUI.LEDOnPulseDur;
    LEDOffDur = S.GUI.OptoFreq - LEDOnPulseDur;
    % PMT shutter signal 5V and 0V durations           
    PMT5VDur = PMTCloseDelay; % set PMT5V dur initially to shutter close delay
    % if the LED is on for longer than the shutter StartOpenDelay,
    % then increase shutter 5V duration by the difference (LEDOnPulseDur - PMTStartOpenDelay)
    if LEDOnPulseDur > PMTStartOpenDelay
        PMT5VDur = PMT5VDur + (LEDOnPulseDur - PMTStartOpenDelay);
    end
    % if shutter duration is less than the minimum dur for the
    % shutter to re-open, set it to minimum shutter pulse dur
    PMT5VDur = max(PMT5VDur, PMTMin5VSignalDur);
    % duration of LED pulse 0V is cycle period minus 5V dur
    PMT0VDur =  S.GUI.OptoFreq - PMT5VDur;
    % integer number of pmt/led cycles within pre + post grating onset
    numPMTLEDCycles = floor(OptoSetup.TimeDur / S.GUI.OptoFreq);
    % LED timers
    sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', LEDOnPulseDur, 'OnsetDelay', LEDOnsetDelay,...
        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
        'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', LEDOffDur,...
        'GlobalTimerEvents', 0, 'OffsetValue', 0);
    % PMT shutter timers
    sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', PMT5VDur, 'OnsetDelay', PMTOnsetDelay,...
        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
        'Loop', numPMTLEDCycles, 'SendGlobalTimerEvents', 0, 'LoopInterval', PMT0VDur,...
        'GlobalTimerEvents', 0, 'OffsetValue', 0);
end


    end
end