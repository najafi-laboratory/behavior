
MaxTrials = 500;
TrialTypes = ceil(rand(1,MaxTrials)*2);

Durs = [];

S.GUI.ISI = 0.75;

S.GUI.Difficulty = 3;

% percentage of full range boundaries for difficulty levels
EasyMinPercent = 2/3;
EasyMaxPercent = 1;
MediumMinPercent = 1/3;
MediumMaxPercent = 2/3;
HardMinPercent = 0;
HardMaxPercent = 1/3;

switch S.GUI.Difficulty % Determine trial-specific visual stimulus
    case 1
        disp('Difficulty - Easy');
    case 2
        disp('Difficulty - Medium');
    case 3
        disp('Difficulty - Hard');
end

%% Main trial loop

for currentTrial = 1:MaxTrials

    % !!! add ISI min/max to param-list later for dynamic video generation
    RandomDurationRangeMin = 70;% Time in ms % Oddball min and max interval duration; uniform distribution.
    %RandomDurationMax = S.GUI.ISI * 1000 - RandomDurationMin; % Time in ms, need to have random duration be at most the ISI, 
    RandomDurationRangeMax = S.GUI.ISI * 1000; % Time in ms, need to have random duration be at most the ISI, 

    RandomDurationFullRange = RandomDurationRangeMax - RandomDurationRangeMin;    

    switch S.GUI.Difficulty % Determine trial-specific visual stimulus
        case 1
            % disp('Difficulty - Easy');
            RandomDurationMin = EasyMinPercent*RandomDurationFullRange;
            RandomDurationMax = EasyMaxPercent*RandomDurationFullRange;
        case 2
            % disp('Difficulty - Medium');
            RandomDurationMin = MediumMinPercent*RandomDurationFullRange;
            RandomDurationMax = MediumMaxPercent*RandomDurationFullRange;
        case 3
            % disp('Difficulty - Hard');
            RandomDurationMin = HardMinPercent*RandomDurationFullRange;
            RandomDurationMax = HardMaxPercent*RandomDurationFullRange;
    end

    % this gives minimum 30ms difference between long/short if drawing max during a 'right' trial and min during a 'left' trial
    RandomDuration = unifrnd(RandomDurationMin, RandomDurationMax, 1, 1)/1000; % get random duration, convert to seconds
    
    % % explicit random duration for debugging video
    %RandomDuration = 1;

    % disp('RandomDuration:');
    % disp(RandomDuration);

    switch TrialTypes(currentTrial) % Determine trial-specific visual stimulus duration
        case 1
            GrayVariableDuration = S.GUI.ISI + RandomDuration; % for left, ISI is added to the random duration
        case 2
            GrayVariableDuration = S.GUI.ISI - RandomDuration; % for right, ISI is subtractd from the random duration
    end  

    Durs =[Durs RandomDuration];
end

X = ['RandomDurationFullRange: ', num2str(RandomDurationFullRange)];
disp(X);
disp('Difficulty Boundaries:');
switch S.GUI.Difficulty % Determine trial-specific visual stimulus
    case 1
        disp(['Easy: Min: ', num2str(RandomDurationMin), ' Max: ', num2str(RandomDurationMax), ' Mean: ', num2str(mean([RandomDurationMax, RandomDurationMin]))]);        
    case 2
        disp(['Medium: Min: ', num2str(RandomDurationMin), ' Max: ', num2str(RandomDurationMax), ' Mean: ', num2str(mean([RandomDurationMax, RandomDurationMin]))]);
    case 3
        disp(['Hard: Min: ', num2str(RandomDurationMin), ' Max: ', num2str(RandomDurationMax), ' Mean: ', num2str(mean([RandomDurationMax, RandomDurationMin]))]);
end
disp(['Draw: Min: ', num2str(min(Durs)), ' Max: ', num2str(max(Durs)), ' Mean: ', num2str(mean(Durs))]);
