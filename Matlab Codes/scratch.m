rng default;  % for reproducibility
MaxTrials = 1000;
TrialTypes = unidrnd(2,1,MaxTrials);

%WarmupTrialsCounter = 5;
WarmupTrialsCounter = MaxTrials;

CurrentTrial = 3;
MaxSameConsecutiveTrials = 3;
SameAsPrevMax = true;

NewTrialTypes = TrialTypes;
for i = CurrentTrial:MaxTrials
    DrawTrialType = unidrnd(2,1,1);
    
    if (i > MaxSameConsecutiveTrials) && (i < WarmupTrialsCounter)
        while SameAsPrevMax
            if ~all(TrialTypes(i-3:i-1) == DrawTrialType)
                SameAsPrevMax = false;
            else
                DrawTrialType = unidrnd(2,1,1);
            end
        end
        NewTrialTypes(i) = DrawTrialType;
    end   
end

figure();
hold on;
plot(TrialTypes, 'o');
%figure();
plot(NewTrialTypes, 'o');
ylim([0 3]);

