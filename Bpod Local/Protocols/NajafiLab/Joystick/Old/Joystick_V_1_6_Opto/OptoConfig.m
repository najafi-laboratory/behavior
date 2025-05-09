classdef OptoConfig
   properties
      EnableOpto = 0;
   end

    methods   

        function obj = OptoConfig(EnableOpto)
            if nargin == 1
                obj.EnableOpto = EnableOpto;
            end
        end

        function obj = set.EnableOpto(obj, EnableOpto)
            obj.EnableOpto = EnableOpto;
        end

        function [OptoTrialTypes] = GenOptoTrials(obj, BpodSystem, S)
            BpodSystem.Data.PreviousSessionType = S.GUI.SessionType;
            BpodSystem.Data.PreviousOptoTrialTypeSeq = S.GUI.OptoTrialTypeSeq;
            BpodSystem.Data.PreviousOnFraction = S.GUI.OnFraction;
            BpodSystem.Data.PreviousNumOptoTrialsPerBlock = S.GUI.NumOptoTrialsPerBlock;
            OptoTrialTypes = ceil(rand(1, S.GUI.MaxTrials)*2);
        end

        function [OptoTrialTypes] = UpdateOptoTrials(obj, BpodSystem, S, OptoTrialTypes, currentTrial, forceUpdate)
            updateOptoTrialTypeSequence = 0;

            if forceUpdate == 1
                updateOptoTrialTypeSequence = 1;
            end
        
            if S.GUI.SessionType == 2
                    S.GUI.OptoTrialTypeSeq = 1;
                    S.GUI.OnFraction = 0;
                    updateOptoTrialTypeSequence = 1;
            end

            if (BpodSystem.Data.PreviousSessionType ~= S.GUI.SessionType) && (S.GUI.SessionType == 1)
                updateOptoTrialTypeSequence = 1;           
            end
            BpodSystem.Data.PreviousSessionType = S.GUI.SessionType;

            if BpodSystem.Data.PreviousOptoTrialTypeSeq ~= S.GUI.OptoTrialTypeSeq
                updateOptoTrialTypeSequence = 1;
            end
            BpodSystem.Data.PreviousOptoTrialTypeSeq = S.GUI.OptoTrialTypeSeq;

            if BpodSystem.Data.PreviousNumOptoTrialsPerBlock ~= S.GUI.NumOptoTrialsPerBlock
                updateOptoTrialTypeSequence = 1;
                BpodSystem.Data.PreviousNumOptoTrialsPerBlock = S.GUI.NumOptoTrialsPerBlock;
            end

            if BpodSystem.Data.PreviousOnFraction ~= S.GUI.OnFraction
                updateOptoTrialTypeSequence = 1;
                BpodSystem.Data.PreviousOnFraction = S.GUI.OnFraction;
            end

            if updateOptoTrialTypeSequence
                numTrialsAddedToSequence = 0;
                OptoTrialTypesToAdd = [];
                switch S.GUI.OptoTrialTypeSeq
                    case 1 % interleaved - random
                        numOnTrialsToAdd = [];
                        OptoTrialTypesToAdd = [repmat(1, 1, S.GUI.MaxTrials - currentTrial + 1)];
                        numOnTrialsToAdd = ceil(S.GUI.OnFraction * length(OptoTrialTypesToAdd));
                        numOnTrialsToAdd_idxs = randperm(length(OptoTrialTypesToAdd), numOnTrialsToAdd);
                        OptoTrialTypesToAdd(numOnTrialsToAdd_idxs) = 2;
                        OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd;
                    case 2 % interleaved - random first trial type
                        FirstTrialType = ceil(rand(1,1)*2);
                        switch FirstTrialType
                            case 1 % on first
                                SecondTrialType = 2;
                            case 2 % off first
                                SecondTrialType = 1;
                        end
                        while numTrialsAddedToSequence < S.GUI.MaxTrials
                            OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(FirstTrialType, 1, S.GUI.NumOptoTrialsPerBlock) repmat(SecondTrialType, 1, S.GUI.NumOptoTrialsPerBlock)];
                            numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                        end
                        OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
                    case 3 % interleaved - short first block
                        while numTrialsAddedToSequence < S.GUI.MaxTrials
                            OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(1, 1, S.GUI.NumOptoTrialsPerBlock) repmat(2, 1, S.GUI.NumOptoTrialsPerBlock)];
                            numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                        end
                        OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
                    case 4 % interleaved - long first block
                        while numTrialsAddedToSequence < S.GUI.MaxTrials
                            OptoTrialTypesToAdd = [OptoTrialTypesToAdd repmat(2, 1, S.GUI.NumOptoTrialsPerBlock) repmat(1, 1, S.GUI.NumOptoTrialsPerBlock)];
                            numTrialsAddedToSequence = numTrialsAddedToSequence + 2*S.GUI.NumOptoTrialsPerBlock;
                        end
                        OptoTrialTypes(currentTrial:end) = OptoTrialTypesToAdd(1:length(OptoTrialTypes) - currentTrial + 1);
                end
            end
        end

        function [AudStimOpto] = GetAudStimOpto(obj, S, OptoTrialType, Vis1Vis2)
            switch obj.EnableOpto
                case 0
                    AudStimOpto = {'HiFi1', ['P', 6]};
                case 1
                    if OptoTrialType == 2
                        switch Vis1Vis2
                            case 1
                                if S.GUI.OptoVis1 && ~S.GUI.OptoWaitForPress1 == 1
                                    AudStimOpto = {'HiFi1', ['P', 6], 'GlobalTimerTrig', '000010001'};
                                elseif S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
                                    AudStimOpto = {'HiFi1', ['P', 6], 'GlobalTimerTrig', '000100001'};
                                else
                                    AudStimOpto = {'HiFi1', ['P', 6]};
                                end
                            case 2
                                if S.GUI.OptoVis2 == 1
                                    AudStimOpto = {'HiFi1', ['P', 6], 'GlobalTimerTrig', '001000100'};
                                elseif S.GUI.OptoVis2 && S.GUI.OptoWaitForPress2
                                    AudStimOpto = {'HiFi1', ['P', 6], 'GlobalTimerTrig', '010000100'};                                    
                                else
                                    AudStimOpto = {'HiFi1', ['P', 6]};
                                end
                        end
                    else
                        AudStimOpto = {'HiFi1', ['P', 6]};
                    end
            end
        end

        function [sma] = InsertGlobalTimer(obj, sma, S, VisStim)
            if obj.EnableOpto 
                % switch S.GUI.PulseType
                %     case 1
                %         Duration = VisStim.VisStimDuration;
                %         % OffDur = VisStim.Grating.Dur;
                %         OffDur = 0;
                %     case 2
                %         T = 1/S.GUI.PulseFreq_Hz;
                %         OnDur = S.GUI.PulseOnDur_ms/1000;
                %         OffDur = abs(OnDur - T);
                %         Duration = OnDur;
                %         LoopInterval = OffDur;
                % end

                % shutter close delays
                PMTStartCloseDelay = 0.010;
                PMTCloseTransferDelay = 0.0121;
                PMTCloseDelay = PMTStartCloseDelay + PMTCloseTransferDelay;

                % shutter open delays
                PMTStartOpenDelay = 0.0078;
                % PMTCloseDur = VisStim.VisStimDuration - PMTStartOpenDelay;
                PMTCloseDur = VisStim.VisStimDuration;
                switch S.GUI.PulseType
                    case 1
                        OnDur = VisStim.VisStimDuration - PMTCloseDelay;
                        OffDur = 0;
                        LoopLED = 0;

                        WaitForPressDur = 30;
                        % WaitForPressLoopLED = 0;
                    case 2
                        T = 1/S.GUI.PulseFreq_Hz;
                        OnDur = S.GUI.PulseOnDur_ms/1000;
                        OffDur = abs(OnDur - T);

                        LoopLED = floor((VisStim.VisStimDuration - PMTCloseDelay)/(OnDur + OffDur));

                        WaitForPressDur = OnDur;
                        % WaitForPressLoopLED = 1;
                end 
              
                % Use separate global timers for each state that opto could
                % be enabled.  Easier to keep track of which need to be
                % triggered and canceled. Also easier to keep
                % trigger/cancel of same timer from both occuring as output
                % of same state.

                % 2 segments of opto for joystick proto - vis1/wait1 and
                % vis2/wait2

                % if first segment only has vis1, then same as 2afc opto
                if S.GUI.OptoVis1
                    % vis 1 led
                    % onset delay of shutter close delay
                    sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', OnDur, 'OnsetDelay', PMTCloseDelay,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', LoopLED, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);

                    % shutter close delay - start close at start of vis
                    % stim, start opening 7.8 ms before LED off
                    sma = SetGlobalTimer(sma, 'TimerID', 5, 'Duration', PMTCloseDur, 'OnsetDelay', 0,...
                        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                        'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);                   
                
                % if first segment has vis1 and/or wait1
                % elseif S.GUI.OptoWaitForPress1
                end
                % wait for press 1 led
                if ~S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
                    %needs onset delay since not preceded
                    % by vis1 LED/shutter
                    sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', WaitForPressDur, 'OnsetDelay', PMTCloseDelay,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);                                       
                    % disp(['Duration OptoWaitForPress1 timer:' num2str(Duration)])
                   
                    % wait1 shutter
                    sma = SetGlobalTimer(sma, 'TimerID', 6, 'Duration', 30, 'OnsetDelay', 0,...
                        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);
                end

                % if contiguous segment 1 (vis1 and wait1)
                if S.GUI.OptoVis1 && S.GUI.OptoWaitForPress1
                    % vis1 led
                    sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', OnDur, 'OnsetDelay', PMTCloseDelay,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);

                    % wait1 led
                    sma = SetGlobalTimer(sma, 'TimerID', 2, 'Duration', WaitForPressDur, 'OnsetDelay', 0,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);      

                    % wait1 shutter
                    sma = SetGlobalTimer(sma, 'TimerID', 6, 'Duration', 30, 'OnsetDelay', 0,...
                        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);
                end

                % vis 2 led
                if S.GUI.OptoVis2
                    % vis 2 led
                    % onset delay of shutter close delay
                    sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', OnDur, 'OnsetDelay', PMTCloseDelay,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', LoopLED, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);


                    % shutter close delay - start close at start of vis
                    % stim, start opening 7.8 ms before LED off
                    sma = SetGlobalTimer(sma, 'TimerID', 7, 'Duration', PMTCloseDur, 'OnsetDelay', 0,...
                        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                        'Loop', 0, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);                        
                end

                % wait for press 2 led
                if ~S.GUI.OptoVis2 && S.GUI.OptoWaitForPress2
                    %needs onset delay since not preceded
                    % by vis1 LED/shutter
                    sma = SetGlobalTimer(sma, 'TimerID', 4, 'Duration', WaitForPressDur, 'OnsetDelay', PMTCloseDelay,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0); 
                    
                    % wait2 shutter
                    sma = SetGlobalTimer(sma, 'TimerID', 8, 'Duration', 30, 'OnsetDelay', 0,...
                        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);
                end

                % if contiguous segment 2
                if S.GUI.OptoVis2 && S.GUI.OptoWaitForPress2
                    % vis2 led
                    sma = SetGlobalTimer(sma, 'TimerID', 3, 'Duration', OnDur, 'OnsetDelay', PMTCloseDelay,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);

                    % wait2 led
                    sma = SetGlobalTimer(sma, 'TimerID', 4, 'Duration', WaitForPressDur, 'OnsetDelay', 0,...
                        'Channel', 'PWM1', 'OnLevel', 255, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);

                    % wait2 shutter
                    sma = SetGlobalTimer(sma, 'TimerID', 8, 'Duration', 30, 'OnsetDelay', 0,...
                        'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                        'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0,...
                        'GlobalTimerEvents', 0, 'OffsetValue', 0);                    
                end                

            end
        end
    end
end
