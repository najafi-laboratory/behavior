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

        function [AudStimOpto] = GetAudStimOpto(obj, OptoTrialType)
            switch obj.EnableOpto
                case 0
                    AudStimOpto = {'HiFi1', ['P', 6]};
                case 1
                    if OptoTrialType == 2
                        AudStimOpto = {'HiFi1', ['P', 6], 'GlobalTimerTrig', 1};
                    else
                        AudStimOpto = {'HiFi1', ['P', 6]};
                    end
            end
        end

        function [sma] = InsertGlobalTimer(obj, sma, S, VisStim)
            if obj.EnableOpto 
                switch S.GUI.PulseType
                    case 1
                        Duration = VisStim.VisStimDuration;
                        % OffDur = VisStim.Grating.Dur;
                        OffDur = 0;
                    case 2
                        T = 1/S.GUI.PulseFreq_Hz;
                        OnDur = S.GUI.PulseOnDur_ms/1000
                        OffDur = abs(OnDur - T);
                        Duration = OnDur;
                        LoopInterval = OffDur;
                end
              
                % sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', VisStim.VisStimDuration, 'OnsetDelay', 0,...
                %     'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                %     'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', VisStim.Grating.Dur,...
                %     'GlobalTimerEvents', 0, 'OffsetValue', 0);
                sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', Duration, 'OnsetDelay', 0,...
                    'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                    'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', OffDur,...
                    'GlobalTimerEvents', 0, 'OffsetValue', 0);                
            end
        end
    end
end
