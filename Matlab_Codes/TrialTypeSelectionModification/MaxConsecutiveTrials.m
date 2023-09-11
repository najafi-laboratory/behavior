%rng default;  % for reproducibility
MaxTrials = 1000;
TrialTypes = unidrnd(2,1,MaxTrials);

%WarmupTrialsCounter = 5;
WarmupTrialsCounter = 20;

%CurrentTrial = 3;
MaxSameConsecutiveTrials = 3;
%NewSameAsPrevMax = true;

NewTrialTypes = TrialTypes;
for i = MaxSameConsecutiveTrials:MaxTrials       
    if (i > MaxSameConsecutiveTrials) && (i <= WarmupTrialsCounter)
        PrevMaxTrials = NewTrialTypes(i-3:i-1);
        % disp(['i:', num2str(i)]);
        if (all(PrevMaxTrials == 1) || all(PrevMaxTrials == 2))
            % disp(['PrevMaxTrials:', num2str(PrevMaxTrials)]);
            NewSameAsPrevMax = true;
            while NewSameAsPrevMax
                DrawTrialType = unidrnd(2,1,1);
                %disp(['i:', num2str(i)]);
                % disp(['NewTrialTypes(i-3:i-1):', num2str(NewTrialTypes(i-3:i-1))]);
                % disp(['DrawTrialType:', num2str(DrawTrialType)]);
                % disp(['all(NewTrialTypes(i-3:i-1) == DrawTrialType):', num2str(all(NewTrialTypes(i-3:i-1) == DrawTrialType))]);         
                if ~all(NewTrialTypes(i-3:i-1) == DrawTrialType)
                    NewSameAsPrevMax = false;
                % else
                %     DrawTrialType = unidrnd(2,1,1);
                end
            end
            NewTrialTypes(i) = DrawTrialType;
        end
    end   
end

figure();
hold on;
plot(TrialTypes, 'o');
%figure();
plot(NewTrialTypes, 'x');
ylim([0 3]);

