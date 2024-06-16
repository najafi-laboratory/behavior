cam = readtable('FN13_20240613_js_t_cam0_run006_20240613_102257.csv');
twop = readtable('FN13_20240613_js_DCNCNO_t-025_Cycle00001_VoltageRecording_001.csv');

cam_time = 1000*cam.timestamp;

tp_volt = twop.Input3;
tp_time = twop.Time_ms_;

tp_onset_inds = find(diff(tp_volt>1)==1)+1;

tp_onset_times = tp_time(tp_onset_inds);


ifi_2p = diff(tp_onset_times);
ifi_cam = diff(cam_time);

figure(); hold on;
plot(ifi_2p,'b'); 
plot(ifi_cam, 'r')

cam_time_r = cam_time - cam_time(1);
tp_onset_times_r = tp_onset_times - tp_onset_times(1);

figure; hold on
plot(cam_time_r)
plot(tp_onset_times_r)



figure; hold on
plot(cam_time_r - tp_onset_times_r)




x_values = tp_onset_times_r;

% Define the y-values for the vertical lines
y_min = 0;  % Minimum y-value for the lines
y_max = 1;  % Maximum y-value for the lines

figure;
hold on;
for i = 1:length(x_values)
    line([x_values(i) x_values(i)], [y_min y_max], 'Color', 'k', 'LineWidth', 1.5);
end
hold off;

xlim([x_values(end)-1000, x_values(end)])
