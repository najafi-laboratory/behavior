trials = SessionData.nTrials;

left_trials = 0;
right_trials = 0;

left_idx = find(SessionData.TrialTypes == 1);
right_idx = find(SessionData.TrialTypes == 2);

num_left = length(left_idx);
num_right = length(right_idx);

labels = {'Reward', ...
    'RewardNaive', ...
    'ChangingMindReward', ...
    'Punish', ...
    'PunishNaive'}

left_counts = zeros(1, 5);
right_counts =zeros(1, 5);

for trial = 1:trials
    for i = 1:length(labels)
        if isfield(SessionData.RawEvents.Trial{1, trial}.States, labels{1, i})            
            if ~isnan(SessionData.RawEvents.Trial{1, trial}.States.(labels{1, i})(1))
                if SessionData.TrialTypes(trial) == 1
                    left_counts(i) = left_counts(i) + 1;
                else
                    right_counts(i) = right_counts(i) + 1;
                end
            end
        end
    end
end

left_percentages = left_counts / sum(left_counts);
right_percentages = right_counts / sum(right_counts);

lick = [];
left_licks = [];
right_licks = [];
correctness = [];
correctness_left = [];
correctness_right = [];
compare = 0; % 0 - neither, 1 - left, 2 - right, 3 - both
for trial = 1:trials
    if isfield(SessionData.RawEvents.Trial{1, trial}.Events, 'Port1In')            
        if ~isnan(SessionData.RawEvents.Trial{1, trial}.Events.Port1In(1))
            % left_licks = [left_licks, SessionData.RawEvents.Trial{1, trial}.Events.Port1In - SessionData.RawEvents.Trial{1, trial}.States.AudStimTrigger(2)];
            % left_licks = [left_licks, SessionData.RawEvents.Trial{1, trial}.Events.Port1In - SessionData.RawEvents.Trial{1, trial}.States.VisStimTrigger(2)];
            left_licks = SessionData.RawEvents.Trial{1, trial}.Events.Port1In;
            left_licks = left_licks(left_licks - SessionData.RawEvents.Trial{1, trial}.States.VisStimTrigger(2) > 0 )
            left_licks = left_licks(left_licks > 0);
            if length(left_licks) > 0
                left_first_lick = left_licks(1);
                if SessionData.TrialTypes(trial) == 1
                    correctness_left = ones(1, length(left_licks));
                else
                    correctness_left = zeros(1, length(left_licks));
                end   
                compare = 1;
            else
                left_licks = [left_licks, NaN];
            end
        else
            left_licks = [left_licks, NaN];
        end
    end
    if isfield(SessionData.RawEvents.Trial{1, trial}.Events, 'Port3In')            
        if ~isnan(SessionData.RawEvents.Trial{1, trial}.Events.Port3In(1))
            % right_licks = [right_licks, SessionData.RawEvents.Trial{1, trial}.Events.Port3In - SessionData.RawEvents.Trial{1, trial}.States.AudStimTrigger(2)];
            % right_licks = [right_licks, SessionData.RawEvents.Trial{1, trial}.Events.Port3In - SessionData.RawEvents.Trial{1, trial}.States.VisStimTrigger(2)];
            right_licks = SessionData.RawEvents.Trial{1, trial}.Events.Port3In;
            right_licks = right_licks(right_licks - SessionData.RawEvents.Trial{1, trial}.States.VisStimTrigger(2) > 0 )
            right_licks = right_licks(right_licks > 0);
            if length(right_licks) > 0            
                right_first_lick = right_licks(1);
                if SessionData.TrialTypes(trial) == 2
                    correctness_right = ones(1, length(right_licks));
                else
                    correctness_right = zeros(1, length(right_licks));
                end      
                compare = compare + 2;
            else
                right_licks = [right_licks, NaN];
            end
        else
            right_licks = [right_licks, NaN];
        end
    end

    switch compare
        case 0
            lick(1, trial) = NaN;
            lick(2, trial) = NaN;
        case 1
            lick(1, trial) = left_first_lick;
            lick(2, trial) = correctness_left(1);
        case 2
            lick(1, trial) = right_first_lick;
            lick(2, trial) = correctness_right(1);
        case 3
            if left_first_lick < right_first_lick
                lick(1, trial) = left_first_lick;
                lick(2, trial) = correctness_left(1);
            else
                lick(1, trial) = right_first_lick;
                lick(2, trial) = correctness_right(1);
            end
    end

    if trial == 100
        disp('bug?')
    end

    left_licks = [];
    right_licks = [];
    correctness = [];
    compare = 0;
end

lick(1,:) = lick(1,:) * 1000;
max_time = 5000

notnan_lick = lick(:,~isnan(lick(1,:)));

% Define bin edges for bin size 250
bin_size = 250;
% bin_edges = 0:bin_size:max(lick(1,:))+bin_size;
bin_edges = 0:bin_size:max_time;

% Discretize the data into bins
bin_indices = discretize(notnan_lick(1,:), bin_edges);

% Display the result
disp('Bin Indices:');
disp(bin_indices);

% Display bin edges
disp('Bin Edges:');
disp(bin_edges);

bin_edges = bin_edges - bin_size / 2;

least_trials = 3;
correctness = [];
bin_mean = [];
bin_sem = [];
bin_time= [];
for i = 1:length(bin_edges)
    correctness = notnan_lick(2, bin_indices == i);   
    n = length(correctness);
    if n > least_trials
        m = mean(correctness);        
        % Sample standard deviation
        s = std(correctness);        
        % Standard error of the mean (SEM)
        SEM = s / sqrt(n);
    else
        m = NaN;
        SEM = NaN;
    end
    bin_mean = [bin_mean, m];
    bin_sem = [bin_sem, SEM];
end

bin_time = bin_edges(1:end-1) + (bin_edges(2)-bin_edges(1)) / 2;
non_nan  = ~isnan(bin_mean);
bin_mean = bin_mean(non_nan);
bin_sem  = bin_sem(non_nan);
bin_time = bin_time(non_nan);

figure()
plot(bin_time, bin_mean)
