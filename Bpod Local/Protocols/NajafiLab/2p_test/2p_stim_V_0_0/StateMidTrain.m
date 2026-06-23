function StateMidTrain(sma, S, SCOA, TrialTarget, VisStimDuration, DURA)
    m_StateUtils = StateUtils;
    sma = m_StateUtils.AddStateInit(sma, S, SCOA);
    sma = m_StateUtils.AddStateStimTrigger(sma, S, SCOA, VisStimDuration);
    sma = m_StateUtils.AddStateChoice(sma, S, SCOA, TrialTarget, DURA, 0, 1);
    sma = m_StateUtils.AddStateReward(sma, S, SCOA, TrialTarget, 0);
    sma = m_StateUtils.AddStatePunish(sma, S, SCOA, 0);
    sma = m_StateUtils.AddStateDidNotChoose(sma, SCOA);
    sma = m_StateUtils.AddStateChangeMind(sma, S, TrialTarget, DURA);
    sma = m_StateUtils.AddStateITI(sma, DURA);
    SendStateMachine(sma);
end