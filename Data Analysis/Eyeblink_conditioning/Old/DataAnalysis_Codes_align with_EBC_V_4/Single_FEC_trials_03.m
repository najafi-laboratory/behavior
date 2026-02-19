clc;close all;clear

data_files = dir('E6LG_EBC_V_4_0_20250605_102813.mat');
% data_files = dir('*_EBC_*.mat');

CR_threshold = 0.05;

good_CR_threshold = 0.05;
poor_CR_threshold = 0.02;

n_row = 4;
n_column = 5;

x_fill_long =[];
x_fill_short=[];
x_fill_shortLED = [];
x_fill_longLED = [];



for ctr_file=1:length(data_files)

load(data_files(ctr_file).name)

delete(strrep(data_files(ctr_file).name, '.mat', '.pdf'))

numTrials = length(SessionData.RawEvents.Trial);
% Create a colormap from light blue to dark blue
colors = [linspace(0, 1, numTrials)', zeros(numTrials, 1), linspace(1, 0, numTrials)'];

    % Variables to track min and max of the curves
    minNumCurve = inf;
    maxNumCurve = -inf;

figure('units','centimeters','position',[1 1 50 30])

for ctr_trial = 1:numTrials


        %  if CheckEyeOpenTimeout is not nan, timeout occurred, continue to
        % next trial
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpenTimeout')
            if ~isnan(SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpenTimeout)
                continue;
            end
        end

        % get CheckEyeOpen if it is in session data for versions V_3_2+
        if isfield(SessionData.RawEvents.Trial{1, ctr_trial}.States, 'CheckEyeOpen')
            CheckEyeOpenStart = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(1);
            CheckEyeOpenStop = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        end

        CheckEyeOpen = SessionData.RawEvents.Trial{1, ctr_trial}.States.CheckEyeOpen(2);
        Start = SessionData.RawEvents.Trial{1, ctr_trial}.States.Start(1);
        ITI_Pre = SessionData.RawEvents.Trial{1, ctr_trial}.States.ITI_Pre(1);
        % LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Onset(1);
          

        block_type = SessionData.RawEvents.Trial{1, ctr_trial}.Data.BlockType;

        if strcmp(block_type, 'warm_up')
            is_warmup = true;
        else
            is_warmup = false;
        end


        
        if SessionData.RawEvents.Trial{1, ctr_trial}.Data.IsProbeTrial
            is_probe = true;
            
        else
           is_probe = false;
        end

        LED_Onset = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_Start;
        LED_Onset_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer1_End;

        if ~SessionData.RawEvents.Trial{1, ctr_trial}.Data.IsProbeTrial  
            LED_Puff_ISI_start = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
            LED_Puff_ISI_end = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2);
            AirPuff_Start = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_Start;
            AirPuff_End = SessionData.RawEvents.Trial{1, ctr_trial}.Events.GlobalTimer2_End;
        
            AirPuff_LED_Onset_Aligned_Start = AirPuff_Start - LED_Onset;
            AirPuff_LED_Onset_Aligned_End = AirPuff_End - LED_Onset;
        
            % Determine short or long
            isLongBlock = LED_Puff_ISI_end - LED_Puff_ISI_start > 0.3;
        else
            % For probe trials, skip airpuff and ISI logic
            AirPuff_LED_Onset_Aligned_Start = NaN;
            AirPuff_LED_Onset_Aligned_End = NaN;
            isLongBlock = NaN;
        end

        FECTimes = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes;   
        FEC_raw = 1 - SessionData.RawEvents.Trial{1, ctr_trial}.Data.eyeAreaPixels ./ SessionData.RawEvents.Trial{1, ctr_trial}.Data.totalEllipsePixels;

        LED_Onset_Zero_Start = LED_Onset - LED_Onset;
        LED_Onset_Zero_End = LED_Onset_End - LED_Onset;

        

        % FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        
        if ~is_probe
            FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Puff_ISI_start;
        else
            FEC_led_aligned = SessionData.RawEvents.Trial{1, ctr_trial}.Data.FECTimes - LED_Onset; % Align to LED only
        end

        fps = 250; % frames per second, frequency of images
        seconds_before = 0.5;
        seconds_after = 2;

        Frames_before = fps * seconds_before;
        Frames_after = fps * seconds_after;

       % Determine a common time vector for interpolation
        common_time_vector = linspace(-seconds_before, seconds_after, Frames_before + Frames_after + 1);

       % Initialize a matrix to store interpolated FEC data
        FEC_norm_matrix = zeros(numTrials, length(common_time_vector));

        abs_FEC_led_aligned = abs(FEC_led_aligned);
        closest_frame_idx_to_LED_Onset = find(abs_FEC_led_aligned == min(abs_FEC_led_aligned));

        start_idx = closest_frame_idx_to_LED_Onset - Frames_before;
        stop_idx = closest_frame_idx_to_LED_Onset + Frames_after;



        len_FEC_led_aligned = length(FEC_led_aligned);
        start_idx = max(1, min(start_idx, len_FEC_led_aligned));
        stop_idx = max(1, min(stop_idx, len_FEC_led_aligned));

        FEC_led_aligned_trimmed = FEC_led_aligned(start_idx : stop_idx);
        FEC_trimmed = FEC_raw(start_idx : stop_idx);
        
        % Apply smoothing to reduce noise
        FEC_led_aligned_trimmed_smooth = smoothdata(FEC_led_aligned_trimmed, 'movmean', 5); % Moving average
        FEC_trimmed_smooth = smoothdata(FEC_trimmed, 'movmean', 5); % Moving average

        
        %isLongBlock = SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(2) - SessionData.RawEvents.Trial{1, ctr_trial}.States.LED_Puff_ISI(1);
       
    


ctr_row = mod(ceil(ctr_trial/n_column)-1,n_row);
ctr_column = mod(ctr_trial-1,n_column);
shrinking_row = 0.9;
shrinking_column = 0.8;



subplot('Position',[0.03+ctr_column/n_column,...
                   0.80-ctr_row/n_row,...
                   shrinking_column/n_column,...
                   shrinking_row/n_column,...
                   ]);


t_LED = LED_Onset_Zero_Start;
t_puff = AirPuff_LED_Onset_Aligned_Start;

        good_CR_threshold = 0.05;
        poor_CR_threshold = 0.02;
% Get CR classification
% CR_category = classify_CR_02(FEC_led_aligned_trimmed, FEC_trimmed, t_LED, t_puff, CR_threshold);
 CR_category = classify_CR_05(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, t_LED, t_puff , good_CR_threshold, poor_CR_threshold);

% Assign color based on classification using switch-case
switch CR_category
    case 'No CR'
        color_str = 'b'; % Blue for No CR
    case 'Poor CR'
        color_str = [1, 0.4, 0.6]; % Pinkish-red for Poor CR
    case 'Good CR'
        color_str = [0.5, 0, 0]; % Deeper Dark Red % Dark Red % Red for Good CR
    otherwise
        color_str = 'k'; % Default color in case of unexpected input
end

plot(FEC_led_aligned_trimmed_smooth,FEC_trimmed_smooth, 'Color', color_str);hold on
    
% Check if there are at least two points to interpolate
if length(FEC_led_aligned_trimmed) > 1 && length(FEC_trimmed) > 1
    % Perform interpolation as usual
    FEC_interpolated = interp1(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, common_time_vector, 'linear', 'extrap');
    FEC_norm_matrix(ctr_trial, :) = FEC_interpolated;

    % Update min and max for the curves
    minNumCurve = min(minNumCurve, min(FEC_interpolated));
    maxNumCurve = max(maxNumCurve, max(FEC_interpolated));
    
    % Plot the interpolated curve
    plot(common_time_vector, FEC_interpolated, 'Color', color_str); hold on
    
else
    % If there are fewer than two points, plot the available point(s) directly
    if length(FEC_led_aligned_trimmed) == 1
        % Only one point is available, plot it as a single point
        plot(FEC_led_aligned_trimmed_smooth, FEC_trimmed_smooth, 'o', 'Color', color_str); hold on
        
        % Update min/max for a single point
        minNumCurve = min(minNumCurve, FEC_trimmed);
        maxNumCurve = max(maxNumCurve, FEC_trimmed);
        
    elseif isempty(FEC_led_aligned_trimmed)
        % If no points are available, just print a message (or handle as needed)
        fprintf('Trial %d has no data points to plot.\n', ctr_trial);
    else
        % If there are a few points, plot them as a line segment (without interpolation)
        plot(FEC_led_aligned_trimmed, FEC_trimmed, 'Color', color_str); hold on
        
        % Update min/max for the available data points
        minNumCurve = min(minNumCurve, min(FEC_trimmed));
        maxNumCurve = max(maxNumCurve, max(FEC_trimmed));
    end
end



        
        if ~is_probe
            y_fill = [0 0 1 1];
            if isLongBlock > 0.3
                x_fill_long = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, ...
                               AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
                fill(x_fill_long, y_fill, [0.5, 1.0, 0.5], 'FaceAlpha', 0.35, 'EdgeColor', 'none');
                fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');
            else
                x_fill_short = [AirPuff_LED_Onset_Aligned_Start, AirPuff_LED_Onset_Aligned_End, ...
                                AirPuff_LED_Onset_Aligned_End, AirPuff_LED_Onset_Aligned_Start];
                x_fill_shortLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
                fill(x_fill_short , y_fill, [0.5, 0.5, 1.0], 'FaceAlpha', 0.15, 'EdgeColor', 'none');
                fill(x_fill_shortLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');
            end

        end

        x_fill_longLED = [LED_Onset_Zero_Start, LED_Onset_Zero_End, LED_Onset_Zero_End, LED_Onset_Zero_Start];
        fill(x_fill_longLED, y_fill, [0.5 0.5 0.5], 'FaceAlpha', 0.16, 'EdgeColor', 'none');

hold off



% Set tick marks to be outside
set(gca, 'TickDir', 'out');
box off
xlim([-0.2 0.6]);
ylim([minNumCurve maxNumCurve]);

set(gca,'FontSize',11)



if is_probe
    title_text = {sprintf('Trial %03.0f {\\bfseries (PROBE)}', ctr_trial)};
else
    title_text = {sprintf('Trial %03.0f', ctr_trial)};
end

if is_warmup
    title_text = {sprintf('Trial %03d {\\bfseries (Warm-up)}', ctr_trial)};
end

title(title_text, 'interpreter', 'latex', 'fontname', 'Times New Roman', 'fontsize', 13);

if is_probe
    ax = gca;
    set(ax, 'Color', [1, 0.9, 0.9]);  % light red/pink background
end

if is_warmup
    ax = gca;
    set(ax, 'Color', [0.95, 0.9, 0.85]);  % light brownish background
end


if((ctr_column==0))
ylabel_text(1) = {'FEC'};
ylabel(ylabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)
end

if((ctr_row==3))
xlabel_text(1) = {'$t\ {\rm{(s)}}$'};
xlabel(xlabel_text,'interpreter', 'latex','fontname','Times New Roman','fontsize',13)
end



if (mod(ctr_trial,20)==0)  || ctr_trial == numTrials  
exportgraphics(gcf,strrep(data_files(ctr_file).name, '.mat', '.pdf'), 'ContentType', 'vector','Append',true);
clf;
% break
end

end
% After finishing the loop, check if there are remaining trials (not divisible by 20)
remainingTrials = mod(ctr_trial, 20);
if remainingTrials > 0
    % Export the remaining trials in the last page of the PDF
    exportgraphics(gcf, strrep(data_files(ctr_file).name, '.mat', '.pdf'), 'ContentType', 'vector', 'Append', true);
    clf;  % Clear figure after exporting
end

end

  