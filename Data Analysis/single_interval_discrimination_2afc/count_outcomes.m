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








