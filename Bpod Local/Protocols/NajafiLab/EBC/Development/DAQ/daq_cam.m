% 
% Data acquisition devices:
% 
% index Vendor Device ID            Description           
% ----- ------ --------- ---------------------------------
% 1     ni     Dev1      National Instruments(TM) USB-6008
% 2     ni     myDAQ1    National Instruments(TM) NI myDAQ



% daq.getDevices

% daqhelp


% ni: National Instruments(TM) NI myDAQ (Device ID: 'myDAQ1')
%    Analog input supports:
%       -2.0 to +2.0 Volts,-10 to +10 Volts ranges
%       Rates from 0.1 to 200000.0 scans/sec
%       2 channels ('ai0','ai1')
%       'Voltage' measurement type
% 
%    Analog output supports:
%       -2.0 to +2.0 Volts,-10 to +10 Volts ranges
%       Rates from 0.1 to 200000.0 scans/sec
%       2 channels ('ao0','ao1')
%       'Voltage' measurement type
% 
%    Digital IO supports:
%       8 channels ('port0/line0' - 'port0/line7')
%       'InputOnly','OutputOnly','Bidirectional' measurement types
% 
%    Counter input supports:
%       Rates from 0.1 to 100000000.0 scans/sec
%       1 channel ('ctr0')
%       'EdgeCount','PulseWidth','Frequency','Position' measurement types
% 
%    Counter output supports:
%       Rates from 0.1 to 100000000.0 scans/sec
%       1 channel ('ctr0')
%       'PulseGeneration' measurement type





% myDAQ

clear all

dq = daq("ni");
flush(dq);
in_ch_0 = addinput(dq, "myDAQ1", "ai0", "Voltage");   % AI channel 0
in_ch_1 = addinput(dq, "myDAQ1", "ai1", "Voltage");   % AI channel 1

% out_ch_0 = addoutput(dq, "myDAQ1", "ao0", "Voltage");   % AI channel 0


dq.Rate = 10000;   % sampling rate
% 
% fs = dq.Rate;   % 10 kS/s
% 
% on_samples  = round(1e-3 * fs);   % 1 ms
% off_samples = round(3e-3 * fs);   % 3 ms
% 
% one_period = [ ...
%     ones(on_samples,1); ...
%     zeros(off_samples,1) ...
% ];
% 
% nPeriods = 250;   % 1 second of data
% signalData = repmat(one_period, nPeriods, 1);
% 
% % Scale pulse amplitude (example: 0–5 V)
% signalData = 5 * signalData;
% 
% write(dq, signalData);

% data = read(dq, seconds(1));
% plot(data.Time, data.ai0)
% start(dq,"Duration",seconds(30))
start(dq,"Duration",hours(2))
% start(dq,"Continuous",hours(2))
% start(dq,"RepeatOutput",hours(2))

dataAll = [];
timestampsAll = [];

% chunkDuration = seconds(1);
chunkDuration = milliseconds(20);
while dq.Running
    [data, timestamps] = read(dq,chunkDuration,OutputFormat="Matrix");
    dataAll = [dataAll; data];
    timestampsAll = [timestampsAll; timestamps];
    % plot(timestampsAll,dataAll)
    plot(timestamps,data)
    pause(0.5*seconds(chunkDuration))
end


% NI USB-6008
% 
% 
% ni: National Instruments(TM) USB-6008 (Device ID: 'Dev1')
%    Analog input supports:
%       8 ranges supported
%       Rates from 0.1 to 10000.0 scans/sec
%       8 channels ('ai0' - 'ai7')
%       'Voltage' measurement type
% 
%    Analog output supports:
%       0 to +5.0 Volts range
%       2 channels ('ao0','ao1')
%       'Voltage' measurement type
% 
%    Digital IO supports:
%       12 channels ('port0/line0' - 'port1/line3')
%       'InputOnly','OutputOnly','Bidirectional' measurement types
% 
%    Counter input supports:
%       1 channel ('ctr0')
%       'EdgeCount' measurement type

% ch_0 = addinput(dq, "Dev1", "port0/line0", "InputOnly");   % DI channel 0
% ch_0 = addinput(dq, "Dev1", "port0/line0", "Digital");   % DI channel 0
% ch_0.TerminalConfig = "SingleEnded";
% ch_1.TerminalConfig = "SingleEnded";
% 
% dq = daq("ni");
% flush(dq);
% ch_0 = addinput(dq, "Dev1", "ai0", "Voltage");   % AI channel 0
% ch_1 = addinput(dq, "Dev1", "ai1", "Voltage");   % AI channel 1
% 
% dq.Rate = 5000;   % sampling rate if 2x channels
% % data = read(dq, seconds(1));
% % plot(data.Time, data.ai0)
% % start(dq,"Duration",seconds(30))
% start(dq,"Duration",hours(2))
% 
% dataAll = [];
% timestampsAll = [];
% 
% % chunkDuration = seconds(1);
% chunkDuration = milliseconds(20);
% while dq.Running
%     % [data, timestamps] = read(dq,chunkDuration,OutputFormat="Matrix");
%     [data, timestamps] = read(dq,"all",OutputFormat="Matrix");
%     data = data';
%     timestamps = timestamps';
%     dataAll = [dataAll data];
%     timestampsAll = [timestampsAll timestamps];
%     % plot(timestampsAll,dataAll)
%     plot(timestamps,data)
%     pause(0.5*seconds(chunkDuration))
% end