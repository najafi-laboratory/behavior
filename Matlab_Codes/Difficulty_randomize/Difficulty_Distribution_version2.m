% define discrete values in distribution
DifficultyLevels = [1, 2, 3];
% assigned probability weights for sampling from distribution
DifficultyProbabilities = [0.5, 0.3, 0.2];

% check that difficulty probability weights sum to 1.0
% maybe change this to message using disp() so it doesn't look like program
% error/crash
AssertCondition = (sum(DifficultyProbabilities) == 1);
AssertErrMsg = ['Sum of difficulty probability weights must equal 1.0'];
assert(AssertCondition, AssertErrMsg);


cp = [0, cumsum(DifficultyProbabilities)]; % cumulative probability -> use as interval to pick Easy OR Medium OR Hard (one occurs every draw)

% for testing this method
resultArray = [];
MaxTrials = 1000;


for currentTrial = 1:MaxTrials
    r = rand; % get random scalar drawn from the uniform distribution in the interval (0,1).
    ind = find(r>cp, 1, 'last');  % get discrete index (1, 2, or 3 for Easy, Medium, or Hard in this case)
    result = DifficultyLevels(ind); % get discrete value at the randomly (according to probability weights) selected index, in this case of 1 = Easy, 2 = Medium, 3 = Hard it will be the same as the index.  This step is here in case more or fewer difficulty levels are added in the future, this gets used as example for drawing from weighted distribution later, or any other unforseen reason
    
    % for testing this method
    resultArray = [resultArray result];
end

% for testing this method
IdxEasy = find(resultArray==1);
IdxMedium = find(resultArray==2);
IdxHard = find(resultArray==3);

NumEasy = length(IdxEasy)
NumMedium = length(IdxMedium)
NumHard = length(IdxHard)

PercentEasy = NumEasy/MaxTrials
PercentMedium = NumMedium/MaxTrials
PercentHard = NumHard/MaxTrials