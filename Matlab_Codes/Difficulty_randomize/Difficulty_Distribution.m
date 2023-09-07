MaxTrials = 500;

PercentEasy = round((2/3) * 100);
PercentMedium = round((1/6) * 100);
PercentHard = round((1/6) * 100);

DifficultyPercentageSum = PercentEasy + PercentMedium + PercentHard; 

% need input validation for bpod GUI
DifficultyPercentageSumTolerance = 0.1;
%DifficultyPercentageSum = 99.5;
if (DifficultyPercentageSum < (100-DifficultyPercentageSumTolerance)) || (DifficultyPercentageSum > (100+DifficultyPercentageSumTolerance))
    disp(['out of tolerance by: ', num2str(abs(abs(DifficultyPercentageSum-100)-DifficultyPercentageSumTolerance))]);
end

NumEasy = floor(PercentEasy/100*MaxTrials);
NumMedium = floor(PercentMedium/100*MaxTrials);
NumHard = floor(PercentHard/100*MaxTrials);

DifficultySum = NumEasy + NumMedium + NumHard;

TrialsWithUnassignedDifficulty = MaxTrials - DifficultySum;
NumEasy = NumEasy + TrialsWithUnassignedDifficulty;
DifficultySum = NumEasy + NumMedium + NumHard;

FinalPercentEasy = (NumEasy / MaxTrials) * 100;
FinalPercentMedium = (NumMedium / MaxTrials) * 100;
FinalPercentHard = (NumHard / MaxTrials) * 100;

disp(['PercentEasy: ', num2str(PercentEasy), ' FinalPercentEasy: ', num2str(FinalPercentEasy)]);
disp(['PercentMedium: ', num2str(PercentMedium), ' FinalPercentMedium: ', num2str(FinalPercentMedium)]);
disp(['PercentHard: ', num2str(PercentHard), ' FinalPercentHard: ', num2str(FinalPercentHard)]);

DiffTypeEasy = 1;
DiffTypeMedium = 2;
DiffTypeHard = 3;

EasyArr = repmat(DiffTypeEasy, 1, NumEasy);
MediumArr = repmat(DiffTypeMedium, 1, NumMedium);
HardArr = repmat(DiffTypeHard, 1, NumHard);

DifficultyArr = [EasyArr MediumArr HardArr];
RandDifficultyArr = DifficultyArr(randperm(length(DifficultyArr)));