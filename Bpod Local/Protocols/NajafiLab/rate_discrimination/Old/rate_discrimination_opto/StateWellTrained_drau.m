function StateWellTrained(S, SCOA, TrialTarget, VisStimDuration, DURA)

    sma = NewStateMatrix();
    sma = SetCondition(sma, 1, TrialTarget.CorrectPort, 1);
    sma = SetCondition(sma, 2, TrialTarget.IncorrectPort, 1);
    sma = SetCondition(sma, 3, 'Port4', 1);
    sma = SetCondition(sma, 4, 'Port4', 0);
    sma = SetCondition(sma, 5, 'Port2', 1);

    sma = SetGlobalTimer(sma, 'TimerID', 1, 'Duration', 0.1, 'OnsetDelay', 0,...
                     'Channel', 'BNC2', 'OnLevel', 1, 'OffLevel', 0,...
                     'Loop', 1, 'SendGlobalTimerEvents', 0, 'LoopInterval', 0);

    sma = AddState(sma, 'Name', 'Start', ...
        'Timer', 0.068,...
        'StateChangeConditions', {'Tup', 'InitCue'},...
        'OutputActions', SCOA.Start);

    switch S.GUI.NoInit
        case 0
            sma = AddState(sma, 'Name', 'InitCue', ...
                'Timer', S.GUI.InitCueDuration_s,...
                'StateChangeConditions', {'Tup', 'InitWindow'},...         
                'OutputActions', SCOA.InitCue);
            sma = AddState(sma, 'Name', 'InitWindow', ...
                'Timer', S.GUI.InitWindowTimeout_s,...
                'StateChangeConditions', { ...
                    'Tup', 'InitCueAgain', ...
                    'Port2In', 'PreVisStimDelay', ...
                    'Port1In', 'WrongInitiation', 'Port3In', 'WrongInitiation'},...
                'OutputActions', {});
            sma = AddState(sma, 'Name', 'InitCueAgain', ...
                'Timer', 0,...
                'StateChangeConditions', {'Tup', 'InitCue'},...         
                'OutputActions', {});
        case 1
            sma = AddState(sma, 'Name', 'InitCue', ...
                'Timer', S.GUI.InitCueDuration_s,...
                'StateChangeConditions', {'Tup', 'PreVisStimDelay'},...         
                'OutputActions', SCOA.InitCue);
    end

    sma = AddState(sma, 'Name', 'WrongInitiation', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...         
        'OutputActions', {});     

    sma = AddState(sma, 'Name', 'PreVisStimDelay', ...
        'Timer', S.GUI.PreVisStimDelay_s,...
        'StateChangeConditions', {'Tup', 'VisStimTrigger'},...
        'OutputActions', {});    

    sma = AddState(sma, 'Name', 'VisStimTrigger', ...
        'Timer', 0,...
        'StateChangeConditions', {'BNC1High', 'AudStimTrigger'},...
        'OutputActions', SCOA.VisStim);

    sma = AddState(sma, 'Name', 'AudStimTrigger', ...
        'Timer', VisStimDuration,...
        'StateChangeConditions', {'PreGoCueDelay'},...
        'OutputActions', SCOA.AudStim);

    sma = AddState(sma, 'Name', 'PreGoCueDelay', ...
        'Timer', S.GUI.PreGoCueDelay_s,...
        'StateChangeConditions', {'Tup', 'GoCue'},...         
        'OutputActions', SCOA.SpoutIn);   

    sma = AddState(sma, 'Name', 'GoCue', ...
        'Timer', S.GUI.GoCueDuration_s,...
        'StateChangeConditions', {'Tup', 'WindowChoice'},...         
        'OutputActions', SCOA.StimAct);      
    
    sma = AddState(sma, 'Name', 'PostRewardDelay', ...
        'Timer', S.GUI.PostRewardDelay_s,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});
    
    sma = AddState(sma, 'Name', 'WindowChoice', ...
        'Timer', DURA.ChoiceWindow,...
        'StateChangeConditions', { ...
            TrialTarget.CorrectLick, 'Reward', ...
            TrialTarget.IncorrectLick, 'PunishSetup', ...
            'Tup', 'DidNotChoose'},...
        'OutputActions', {});      

    sma = AddState(sma, 'Name', 'DidNotChoose', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', {});   

    sma = AddState(sma, 'Name', 'Reward', ...
        'Timer', TrialTarget.ValveTime,...
        'StateChangeConditions', {'Tup', 'PostRewardDelay'},...
        'OutputActions', {TrialTarget.Valve, 1});
  
    sma = AddState(sma, 'Name', 'PunishSetup', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'Punish'},...
        'OutputActions', SCOA.Punish);

    sma = AddState(sma, 'Name', 'Punish', ...
        'Timer', S.GUI.NoiseDuration_s,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', SCOA.Punish);

    sma = AddState(sma, 'Name', 'TimeOutPunish', ...
        'Timer', DURA.TimeOutPunish,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'ITI', ...
        'Timer', DURA.ITI,...
        'StateChangeConditions', {'Tup', '>exit'},...
        'OutputActions', {'SoftCode', 254, 'HiFi1', 'X'});

    SendStateMachine(sma);
end