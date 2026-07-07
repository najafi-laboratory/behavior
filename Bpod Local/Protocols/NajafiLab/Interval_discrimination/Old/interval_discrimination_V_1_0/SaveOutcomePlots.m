function SaveOutcomePlots
global BpodSystem
SessionData = BpodSystem.Data;
%save(BpodSystem.Path.CurrentDataFile, 'SessionData');  % store in orginal file path

% get pre-set file path BpodSystem.Path.CurrentDataFile
%disp(['BpodSystem.Path.CurrentDataFile:', BpodSystem.Path.CurrentDataFile]);
% split file path into primary parts
[Filepath, Name, Ext] = fileparts(BpodSystem.Path.CurrentDataFile);
% split directory path
Filepathologyparts = strsplit(Filepath, filesep);

% remove protocol folder from path
AlternateFilePath = [Filepathologyparts(1:(end-2)) Filepathologyparts(end)];

% reconstruct file path
AlternateCurrentDataFileDir = '';
AlternateCurrentDataFileDir = [AlternateFilePath{1, 1}]; % add drive
%for i = 2:(length(AlternateFilePath)-1) % exclude 'Session Data' folder
% for i = 2:3 % exclude 'Session Data' folder
%     AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep AlternateFilePath{1, i}];
% end

AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep AlternateFilePath{1, 2}];    % behavior dir
AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep 'session_data']; % session_data
AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep AlternateFilePath{1, 6}];    % subject name

% make data folder for current test subject if it doesn't already
% exist
[status, msg, msgID] = mkdir(AlternateCurrentDataFileDir);

% add filename and extension to path
AlternateCurrentDataFile = fullfile(AlternateCurrentDataFileDir, [Name, Ext]);

exportgraphics(BpodSystem.GUIHandles.OutcomePlot, 'plot2.pdf', 'ContentType', 'vector', 'Append', true);