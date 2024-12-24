%{
----------------------------------------------------------------------------

This file is part of the Sanworks Bpod repository
Copyright (C) 2019 Sanworks LLC, Stony Brook, New York, USA

----------------------------------------------------------------------------

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3.

This program is distributed  WITHOUT ANY WARRANTY and without even the 
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
%}
function SaveBpodSessionData
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

switch BpodSystem.Data.RigName
    case '2AFCRig1'
        AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep AlternateFilePath{1, 2} ...
            filesep AlternateFilePath{1, 3}];    % behavior dir
        AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep 'session_data']; % session_data
        AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep AlternateFilePath{1, 7}];    % subject name
    otherwise        
        AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep AlternateFilePath{1, 2}];    % behavior dir
        AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep 'session_data']; % session_data
        AlternateCurrentDataFileDir = [AlternateCurrentDataFileDir filesep AlternateFilePath{1, 6}];    % subject name
end

% make data folder for current test subject if it doesn't already
% exist
[status, msg, msgID] = mkdir(AlternateCurrentDataFileDir);

% add filename and extension to path
AlternateCurrentDataFile = fullfile(AlternateCurrentDataFileDir, [Name, Ext]);

% save session file
save(AlternateCurrentDataFile, 'SessionData');


