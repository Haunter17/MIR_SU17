function createDatasetExp7e(filelist, downsamplingRate, cqtDir)

% ***Note: This is a new version of createDatasetBatch which SEPARATES
% features and labels before saving the matrices. The resulting feature 
% matrices will have the dimension of 121 x number of samples and the 
% resulting labels 1 x number of samples. These matrices can be read by 
% python code that is not marked (old).***

% Call pitchProfilePartial to compute energy of each pitch class for 
% each track in filelist. This is for test files, because training is done
% only on songs without pitch shifts.

% create the CQT files

% add path to ComputeQSpecBatch
addpath('../../')

computeQSpecBatch(filelist, cqtDir);

fid = fopen(filelist);
curfile = fgetl(fid);
trainingSet = [];
testSet = [];
label = 0;

newFileNamesID = fopen('CQTFilenames.list', 'w')

while ischar(curfile)
	disp(['-- Generating data matrix on file ', int2str(label)]);
    [pathstr,name,ext] = fileparts(curfile);
    cqt_file = strcat(cqtdir,name,'.mat');
    Qfile = load(cqt_file); % loads Q (struct)
    % preprocessing of Q: cubic root
    Q_Mat = nthroot(abs(Qfile.Q.c), 3);
    % average every 15 columns
    QMat = avg_kcol(Q_Mat, downsamplingRate);

    % save the matrix of CQT coefficients
    [filePath, fileName, fileExt] = fileparts(curfile);
    CQTFilename = strcat(fileName, '_CQT', '.mat');
    save(CQTFilename, 'QMat', '-v7.3');
    % also save the filename to a list
    fprintf(newFileNamesID, '%s\n', CQTFilename);

    curfile = fgetl(fid);
    label = label + 1;
end

end

function [ D ] = avg_kcol( A, k )
% average a matrix by every k columns
%   
[orig_row, orig_col] = size(A);
new_col = floor(orig_col / k) * k;
A = A(:, 1: new_col);
B = reshape(A', k, [])';
C = mean(B, 2);
D = reshape(C', [], orig_row)';
end

function [ D ] = reshape_prow( A, p )
% reshape the matrix to have p x original number of rows
[orig_row, orig_col] = size(A);
new_col = floor(orig_col / p) * p;
A = A(:, 1 : new_col);
B = reshape(A', [], orig_row * p)';
idx = [];
for i = 1 : p
    curr_idx = [i : p : orig_row * p];
    idx = [idx curr_idx];
end
D = B(idx, :);
end
