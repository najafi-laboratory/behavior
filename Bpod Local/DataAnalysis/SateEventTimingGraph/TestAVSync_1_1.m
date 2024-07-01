


F1_samples = [];
F23_samples = [];
Vis1AsMeasuredByBNC_samples = [];
Grating_samples = [];
Stim_BNC_Diff_samples = [];

VisDetectDur_samples = [];

AudioDurationAsMeasuredByBNC_samples = [];
AudioStartOffsetFromF23_samples = [];

missing_state_offset = 0;

for trial = 1:SessionData.nTrials
    
    % trial = 5; % test individual trial

    Events = SessionData.RawEvents.Trial{1, trial}.Events;
    States = SessionData.RawEvents.Trial{1, trial}.States;
    
    if ~isnan(States.VisStimInterrupt(1))
        missing_state_offset = missing_state_offset + 1;
        continue;
    end
    % if (trial == 56) %|| (trial == 4)
    %     missing_state_offset = missing_state_offset + 1;
    %     continue;
    % end

    %% BNC1
    % BNC1High for trial
    BNC1High = Events.BNC1High;
    % BNC1Low for trial
    BNC1Low = Events.BNC1Low;

    %% BNC2

    % BNC2High for trial
    BNC2High = Events.BNC2High;
    % BNC2Low for trial
    BNC2Low = Events.BNC2Low;    

    %% Vis1
    % First gray frame
    F1 = BNC1Low(1) - BNC1High(1);
    % Second gray frame
    F23 = BNC1High(2) - BNC1Low(1);
    % BNC measured duration of gray frames
    GrayShift = F23;
    % BNC measured duration of grating frames
    Grating = BNC1Low(2) - BNC1High(2);
    % BNC measured duration of Vis1
    Vis1AsMeasuredByBNC = GrayShift + Grating;
    
    %% Vis Detect
    VisDetect1 = States.VisDetect1;
    VisDetectDur = VisDetect1(2) - VisDetect1(1);

    % AV
    AudioDurationAsMeasuredByBNC = BNC2Low(1) - BNC2High(1);
    AudioStartOffsetFromF23 = BNC2High(1) - BNC1Low(1);

    disp(['F1 ' num2str(F1)]);
    disp(['F23 ' num2str(F23)]);
    disp(['GrayShift ' num2str(GrayShift)]);
    disp(['Grating ' num2str(Grating)]);
    disp(['Vis1AsMeasuredByBNC ' num2str(Vis1AsMeasuredByBNC)]);
    disp(['VisDetectDur ' num2str(VisDetectDur)]);
    
    VisualStimulus1 = States.VisualStimulus1;
    % State timer measured duration of VisualStimulus1 (state duration)
    Stim1Dur = VisualStimulus1(2) - VisualStimulus1(1);
    disp(['Stim1Dur ' num2str(Stim1Dur)]);
    
    % Difference between state timer and BNC measure of visual stimulus
    Stim_BNC_Diff = Stim1Dur - Vis1AsMeasuredByBNC;    
    disp(['Stim_BNC_Diff ' num2str(Stim_BNC_Diff)]);
    
    % Difference between photodiode detection of first frame and start of
    % visual stimulus state
    BNC_vis_start_diff = BNC1High(1) - VisualStimulus1(1);
    % Difference between photodiode detection of background gray frame and
    % end of visual stimulus state
    BNC_vis_stop_diff = BNC1Low(2) - VisualStimulus1(2);
    disp(['BNC_vis_start_diff ' num2str(BNC_vis_start_diff)]);
    disp(['BNC_vis_stop_diff ' num2str(BNC_vis_stop_diff)]);
        
    VisualStimulus2 = States.VisualStimulus2;

    % stat samples
    F1_samples = [F1_samples F1];
    F23_samples = [F23_samples F23];
    Vis1AsMeasuredByBNC_samples = [Vis1AsMeasuredByBNC_samples Vis1AsMeasuredByBNC];
    Grating_samples = [Grating_samples Grating];
    Stim_BNC_Diff_samples= [Stim_BNC_Diff_samples Stim_BNC_Diff];
    VisDetectDur_samples = [VisDetectDur_samples VisDetectDur];

    AudioDurationAsMeasuredByBNC_samples = [AudioDurationAsMeasuredByBNC_samples AudioDurationAsMeasuredByBNC];
    AudioStartOffsetFromF23_samples = [AudioStartOffsetFromF23_samples AudioStartOffsetFromF23];
end

F1_samples = SToMillis(F1_samples);
F23_samples = SToMillis(F23_samples);
Vis1AsMeasuredByBNC_samples = SToMillis(Vis1AsMeasuredByBNC_samples);
Grating_samples = SToMillis(Grating_samples);
Stim_BNC_Diff_samples = SToMillis(Stim_BNC_Diff_samples);
VisDetectDur_samples = SToMillis(VisDetectDur_samples);
AudioDurationAsMeasuredByBNC_samples = SToMillis(AudioDurationAsMeasuredByBNC_samples);
AudioStartOffsetFromF23_samples = SToMillis(AudioStartOffsetFromF23_samples);

x = [1:SessionData.nTrials - missing_state_offset]';

figure
boxplot([F1_samples', F23_samples', Vis1AsMeasuredByBNC_samples', Grating_samples', Stim_BNC_Diff_samples', AudioDurationAsMeasuredByBNC_samples', AudioStartOffsetFromF23_samples', VisDetectDur_samples'],...
    'Labels',{'Frame 1','Frame 2,3','Vis1 BNC','Grating','Vis1StateDur - Vis1 BNC','AudioDur BNC','AudioStartOffsetFromF23', 'VisDetectDur'})
title('Compare State and Event Timing')

figure
hold on;
plot(x, F1_samples)
plot(x, F23_samples)
plot(x, Vis1AsMeasuredByBNC_samples)
plot(x, Grating_samples)
plot(x, Stim_BNC_Diff_samples)
plot(x, AudioDurationAsMeasuredByBNC_samples)
plot(x, AudioStartOffsetFromF23_samples)
plot(x, VisDetectDur_samples)

title('State and Event Sync/Timing')
xlabel('Trial Number') 
ylabel('Time (ms)')
legend({'Frame 1','Frame 2,3','Vis1 BNC','Grating','Vis1StateDur - Vis1 BNC',...
    'AudioDur BNC','AudioStartOffsetFromF23', 'VisDetectDur'})

disp([' ']);

disp(['max(F1_samples) ' num2str(max(abs(F1_samples)))]);
disp(['min(F1_samples) ' num2str(min(abs(F1_samples)))]);
disp(['mean(F1_samples) ' num2str(mean(abs(F1_samples)))]);
disp(['std(F1_samples) ' num2str(std(abs(F1_samples)))]);
disp(['var(F1_samples) ' num2str(var(abs(F1_samples)))]); 

disp([' ']);

disp(['max(F23_samples) ' num2str(max(abs(F23_samples)))]);
disp(['min(F23_samples) ' num2str(min(abs(F23_samples)))]);
disp(['mean(F23_samples) ' num2str(mean(abs(F23_samples)))]);
disp(['std(F23_samples) ' num2str(std(abs(F23_samples)))]);
disp(['var(F23_samples) ' num2str(var(abs(F23_samples)))]);  

disp([' ']);

disp(['max(Vis1AsMeasuredByBNC_samples) ' num2str(max(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['min(Vis1AsMeasuredByBNC_samples) ' num2str(min(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['mean(Vis1AsMeasuredByBNC_samples) ' num2str(mean(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['std(Vis1AsMeasuredByBNC_samples) ' num2str(std(abs(Vis1AsMeasuredByBNC_samples)))]);
disp(['var(Vis1AsMeasuredByBNC_samples) ' num2str(var(abs(Vis1AsMeasuredByBNC_samples)))]);   

disp([' ']);

disp(['max(Grating_samples) ' num2str(max(abs(Grating_samples)))]);
disp(['min(Grating_samples) ' num2str(min(abs(Grating_samples)))]);
disp(['mean(Grating_samples) ' num2str(mean(abs(Grating_samples)))]);
disp(['std(Grating_samples) ' num2str(std(abs(Grating_samples)))]);
disp(['var(Grating_samples) ' num2str(var(abs(Grating_samples)))]); 

disp([' ']);

disp(['max(Stim_BNC_Diff_samples) ' num2str(max(abs(Stim_BNC_Diff_samples)))]);
disp(['min(Stim_BNC_Diff_samples) ' num2str(min(abs(Stim_BNC_Diff_samples)))]);
disp(['mean(Stim_BNC_Diff_samples) ' num2str(mean(abs(Stim_BNC_Diff_samples)))]);
disp(['std(Stim_BNC_Diff_samples) ' num2str(std(abs(Stim_BNC_Diff_samples)))]);
disp(['var(Stim_BNC_Diff_samples) ' num2str(var(abs(Stim_BNC_Diff_samples)))]);

disp([' ']);

disp(['max(AudioDurationAsMeasuredByBNC_samples) ' num2str(max(abs(AudioDurationAsMeasuredByBNC_samples)))]);
disp(['min(AudioDurationAsMeasuredByBNC_samples) ' num2str(min(abs(AudioDurationAsMeasuredByBNC_samples)))]);
disp(['mean(AudioDurationAsMeasuredByBNC_samples) ' num2str(mean(abs(AudioDurationAsMeasuredByBNC_samples)))]);
disp(['std(AudioDurationAsMeasuredByBNC_samples) ' num2str(std(abs(AudioDurationAsMeasuredByBNC_samples)))]);
disp(['var(AudioDurationAsMeasuredByBNC_samples) ' num2str(var(abs(AudioDurationAsMeasuredByBNC_samples)))]); 

disp([' ']);

disp(['max(AudioStartOffsetFromF23_samples) ' num2str(max(abs(AudioStartOffsetFromF23_samples)))]);
disp(['min(AudioStartOffsetFromF23_samples) ' num2str(min(abs(AudioStartOffsetFromF23_samples)))]);
disp(['mean(AudioStartOffsetFromF23_samples) ' num2str(mean(abs(AudioStartOffsetFromF23_samples)))]);
disp(['std(AudioStartOffsetFromF23_samples) ' num2str(std(abs(AudioStartOffsetFromF23_samples)))]);
disp(['var(AudioStartOffsetFromF23_samples) ' num2str(var(abs(AudioStartOffsetFromF23_samples)))]); 

disp([' ']);

disp(['max(VisDetectDur_samples) ' num2str(max(abs(VisDetectDur_samples)))]);
disp(['min(VisDetectDur_samples) ' num2str(min(abs(VisDetectDur_samples)))]);
disp(['mean(VisDetectDur_samples) ' num2str(mean(abs(VisDetectDur_samples)))]);
disp(['std(VisDetectDur_samples) ' num2str(std(abs(VisDetectDur_samples)))]);
disp(['var(VisDetectDur_samples) ' num2str(var(abs(VisDetectDur_samples)))]);
    
% figure('Name', 'F1_samples', 'NumberTitle', 'off')
% histogram(F1_samples)
% figure('Name', 'F23_samples', 'NumberTitle', 'off')
% histogram(F23_samples)
% figure('Name', 'Vis1AsMeasuredByBNC_samples', 'NumberTitle', 'off')
% histogram(Vis1AsMeasuredByBNC_samples)
% figure('Name', 'Grating_samples', 'NumberTitle', 'off')
% histogram(Grating_samples)
% figure('Name', 'Stim_BNC_Diff_samples', 'NumberTitle', 'off')
% histogram(Stim_BNC_Diff_samples)




function [dataInMillis] = SToMillis(array)
    dataInMillis = array * 1000;
end
