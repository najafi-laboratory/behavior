function StateMidTrain(sma, S, SCOA, TrialTarget, VisStimDuration, DURA)

    sma = SetCondition(sma, 1, TrialTarget.CorrectPort, 1);
    sma = SetCondition(sma, 2, TrialTarget.IncorrectPort, 1);
    sma = SetCondition(sma, 3, 'Port4', 1);
    sma = SetCondition(sma, 4, 'Port4', 0);
    sma = SetCondition(sma, 5, 'Port2', 1);

    sma = AddState(sma, 'Name', 'Start', ...
        'Timer', 0.068,...
        'StateChangeConditions', {'Tup', 'PreVisStimDelay'},...
        'OutputActions', SCOA.Start);  

    sma = AddState(sma, 'Name', 'PreVisStimDelay', ...
        'Timer', S.GUI.PreVisStimDelayMin_s,...
        'StateChangeConditions', {'Tup', 'VisStimTrigger'},...
        'OutputActions', {});    

    sma = AddState(sma, 'Name', 'VisStimTrigger', ...
        'Timer', 0,...
        'StateChangeConditions', {'BNC1High', 'AudStimTrigger'},...
        'OutputActions', SCOA.VisStim);

    sma = AddState(sma, 'Name', 'AudStimTrigger', ...
        'Timer', VisStimDuration + S.GUI.ChoiseWindowStartDelay,...
        'StateChangeConditions', {'Tup', 'PreGoCueDelay'},...
        'OutputActions', SCOA.AudStim);

    sma = AddState(sma, 'Name', 'PostVisStimDelay', ...
        'Timer', DURA.PostVisStimDelay,...
        'StateChangeConditions', {'Tup', 'PreGoCueDelay'},...
        'OutputActions', SCOA.AudStim);    

    sma = AddState(sma, 'Name', 'PreGoCueDelay', ...
        'Timer', S.GUI.PreGoCueDelay_s,...
        'StateChangeConditions', {'Tup', 'GoCue'},...
        'OutputActions', SCOA.SpoutIn);   

    sma = AddState(sma, 'Name', 'GoCue', ...
        'Timer', S.GUI.GoCueDuration_s,...
        'StateChangeConditions', {'Tup', 'WindowChoice'},...         
        'OutputActions', SCOA.StimAct);      
    
    sma = AddState(sma, 'Name', 'WindowChoice', ...
        'Timer', DURA.ChoiceWindow,...
        'StateChangeConditions', { ...
            TrialTarget.CorrectLick, 'RewardSetup', ...
            TrialTarget.IncorrectLick, 'PunishSetup', ...
            'Tup', 'DidNotChoose'},...
        'OutputActions', {});      
    
    sma = AddState(sma, 'Name', 'RewardSetup', ...
        'Timer', S.GUI.OutcomeFeedbackDelay,...
        'StateChangeConditions', {'Tup', 'PreRewardDelay'},...
        'OutputActions', {'GlobalTimerCancel', '11'});    

    sma = AddState(sma, 'Name', 'PreRewardDelay', ...
        'Timer', S.GUI.PreRewardDelay,...
        'StateChangeConditions', {'Tup', 'Reward'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'Reward', ...
        'Timer', TrialTarget.ValveTime,...
        'StateChangeConditions', {'Tup', 'PostRewardDelay'},...
        'OutputActions', {TrialTarget.Valve, 1});

    sma = AddState(sma, 'Name', 'PostRewardDelay', ...
        'Timer', S.GUI.PostRewardDelay,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});   

    sma = AddState(sma, 'Name', 'DidNotChoose', ...
        'Timer', 0,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', {'GlobalTimerCancel', '11'});      
  
    sma = AddState(sma, 'Name', 'PunishSetup', ...
        'Timer', S.GUI.OutcomeFeedbackDelay,...
        'StateChangeConditions', {'Tup', 'Punish'},...
        'OutputActions', {'SoftCode', 254});

    sma = AddState(sma, 'Name', 'Punish', ...
        'Timer', S.GUI.NoiseDuration_s,...
        'StateChangeConditions', {'Tup', 'PostPunishDelay'},...
        'OutputActions', SCOA.Punish);

    sma = AddState(sma, 'Name', 'PostPunishDelay', ...
        'Timer', S.GUI.PostOutcomeDelay,...
        'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
        'OutputActions', {});
   
    sma = AddState(sma, 'Name', 'TimeOutPunish', ...
        'Timer', DURA.TimeOutPunish,...
        'StateChangeConditions', {'Tup', 'ITI'},...
        'OutputActions', {});

    sma = AddState(sma, 'Name', 'ITI', ...
        'Timer', DURA.ITI,...
        'StateChangeConditions', {'Tup', '>exit'},...
        'OutputActions', {'SoftCode', 254, 'HiFi1', 'X'});

    SendStateMachine(sma);

    % sma = AddState(sma, 'Name', 'RewardSetup', ...
    %     'Timer', S.GUI.OutcomeFeedbackDelay,...
    %     'StateChangeConditions', {'Tup', 'Reward'},...
    %     'OutputActions', {'GlobalTimerCancel', '11'});

    % sma = AddState(sma, 'Name', 'ChangingMindWindow', ...
    %     'Timer', DURA.ChangeMindDur,...
    %     'StateChangeConditions', {TrialTarget.CorrectLick, 'ChangingMindReward', 'Tup', 'PunishSetup'},...
    %     'OutputActions', {'GlobalTimerCancel', '11'});
    % 
    % sma = AddState(sma, 'Name', 'ChangingMindReward', ...
    %     'Timer', 0,...
    %     'StateChangeConditions', {'Tup', 'PostChangingMindRewardDelay'},...
    %     'OutputActions', {TrialTarget.Valve, 1});
    % 
    % sma = AddState(sma, 'Name', 'PostChangingMindRewardDelay', ...
    %     'Timer', S.GUI.PostRewardDelay,...
    %     'StateChangeConditions', {'Tup', 'TimeOutPunish'},...
    %     'OutputActions', {});
end