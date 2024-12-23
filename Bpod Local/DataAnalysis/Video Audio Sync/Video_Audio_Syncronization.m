disp('----------------------------------------------------------');

numTrials = 5;

StimulusDuration = 0.1;
ISI = 0.5;

for Trial_num = 1:numTrials
    %Trial_num = 3;
    
    % Frame2TTL Sync signals
    BNC1Low_x = SessionData.RawEvents.Trial{1, Trial_num}.Events.BNC1Low;
    BNC1Low_y = zeros(1, length(BNC1Low_x));
    
    BNC1High_x = SessionData.RawEvents.Trial{1, Trial_num}.Events.BNC1High;
    BNC1High_y = ones(1, length(BNC1High_x));

    % Hifi audio Sync signals
    BNC2Low_x = SessionData.RawEvents.Trial{1, Trial_num}.Events.BNC2Low;
    BNC2Low_y = ones(1, length(BNC2Low_x));

    BNC2High_x = SessionData.RawEvents.Trial{1, Trial_num}.Events.BNC2High;
    BNC2High_y = ones(1, length(BNC2High_x));        

    % 

    % figure();
    % hold on;
    % plot(BNC1Low_x,BNC1Low_y,'o');
    % plot(BNC1High_x,BNC1High_y,'o');
    % %xlim([0 10]);
    % ylim([-0.5 1.5]);
    
    % concatenate time arrays of low and high sync signals
    t = [BNC1Low_x BNC1High_x];
    % concatenate logic value arrays of low and high sync signals
    v = [BNC1Low_y BNC1High_y];
    
    % sort t in ascending order (increasing t values) 
    % and keep the sort index in "sortIdx"
    [t,sortIdx] = sort(t,'ascend');
    % sort v using the sorting index
    v = v(sortIdx);
    
    % figure();
    % hold on;
    % % plot(BNC1Low_x,BNC1Low_y,'o');
    % % plot(BNC1High_x,BNC1High_y,'o');
    % plot(t,v);
    % plot(t,v,'o');
    % %xlim([0 10]);
    % ylim([-0.5 1.5]);
    
    % define query points for interpolation using 'previous' method to generate
    % square wave which should be accurate to Frame2TTL's photodiode input
    t_start = t(1);
    t_end = t(length(t));
    t_interval = t_end - t_start;
    t_increment = t_interval / (length(t) * 20);
    tq = t_start:t_increment:t_end;
    v_interp = interp1(t,v,tq,'previous');
    
    % vis stim state    
    t_VisStim = SessionData.RawEvents.Trial{1, Trial_num}.States.VisualStimulus;
    VisStimStateStart = t_VisStim(1);
    VisStimStateEnd = t_VisStim(end);
    SyncPatchStart = t(1);
    TimeBetweenState_and_Sync = SyncPatchStart - VisStimStateStart;
    disp(['Vis Stim State start: ', num2str(VisStimStateStart)]);
    disp(['Vis Stim State end: ', num2str(VisStimStateEnd)]);
    disp(['Frame2TTL first displayed sync patch: ', num2str(SyncPatchStart)]);
    disp(['Time between start of vis stim state and first sync patch: ', num2str(TimeBetweenState_and_Sync)]); 
    TimeBetweenVisStimStateEndAndGoCueBNCSync = BNC2High_x(end) - VisStimStateEnd;
    disp(['Time between end of vis stim state and go cue BNC sync: ', num2str(TimeBetweenVisStimStateEndAndGoCueBNCSync)]);

    % go cue state
    t_GoCue = SessionData.RawEvents.Trial{1, Trial_num}.States.GoCue;
    GoCueStateStart = t_GoCue(1);
    TimeBetweenVisStimStateEndAndGoCueStateStart = GoCueStateStart - VisStimStateEnd;
    disp(['Time between end of vis stim state and start of go cue state: ', num2str(TimeBetweenVisStimStateEndAndGoCueStateStart)]);
    
    % find average frequency of sync patch as displayed'x'
    t_diff = diff(t);
    f_avg = 1 / mean(t_diff);
    disp(['Average Frame2TTL signal freq:', num2str(f_avg)]);

    VisStimBoundaries = [SyncPatchStart,
    SyncPatchStart + StimulusDuration,
    SyncPatchStart + StimulusDuration + ISI,
    SyncPatchStart + 2*StimulusDuration + ISI,
    SyncPatchStart + 2*StimulusDuration + 2*ISI,
    SyncPatchStart + 3*StimulusDuration + 2*ISI,
    SyncPatchStart + 3*StimulusDuration + 3*ISI,
    SyncPatchStart];

    
    figure();
    hold on;
    % plot(BNC1Low_x,BNC1Low_y,'o');
    % plot(BNC1High_x,BNC1High_y,'o');
    %plot(t,v);
    plot(tq, v_interp);
    plot(t,v,'o');
    plot(VisStimStateStart, v(1), 'x'); % show start of vis stim state (play video is output action of this state)
    plot(VisStimStateEnd, v(1), 'pentagram');  % end of vis stim state
    plot(GoCueStateStart, v(1), 'hexagram');  % end of vis stim state
    xline(VisStimBoundaries);
    % plot audio
    plot(BNC2High_x, BNC2High_y, '*');

    %xlim([0 10]);
    ylim([-0.5 1.5]);
end
