function createDatasetBatch(filelist, filename, downsamplingRate, outdir)

% ***Note: This is a new version of createDatasetBatch which SEPARATES
% features and labels before saving the matrices. The resulting feature 
% matrices will have the dimension of 121 x number of samples and the 
% resulting labels 1 x number of samples. These matrices can be read by 
% python code that is not marked (old).***

% Call pitchProfilePartial to compute energy of each pitch class for 
% each track in filelist. This is for test files, because training is done
% only on songs without pitch shifts.

fid = fopen(filelist);
curfile = fgetl(fid);
trainingSet = [];
validationSet = [];
label = 1;

while ischar(curfile)
    disp(['-- Generating data matrix on file ', int2str(label)]);
    [pathstr,name,ext] = fileparts(curfile);
    cqt_file = strcat(outdir,name,'.mat');
    Qfile = load(cqt_file); % loads Q (struct)
    % preprocessing of Q: cubic root
    Q_Mat = nthroot(abs(Qfile.Q.c), 3);
    % average every 15 columns
    % QMat = avg_kcol(Q_Mat, downsamplingRate);
    % subsampling
    QMat = Q_Mat(:,1:downsamplingRate:size(Q_Mat,2));

    num_frame_agg = 32;
    strideSize = 8;

    [trainVec, validationVec] = createDataset(QMat, label, strideSize, num_frame_agg);
    trainingSet = cat(1, trainingSet, trainVec(:,1:3840));
    validationSet = cat(1, validationSet, validationVec(:,1:1920));

    curfile = fgetl(fid);
    
    label = label + 1;
end
 
disp(['==> Shuffling data...'])
trainingSet = trainingSet(randperm(size(trainingSet, 1)), :)';
validationSet =validationSet(randperm(size(validationSet, 1)), :)';

disp(['==> Separate coefficients and labels'])
trainingLabels = trainingSet(size(trainingSet,1), :);
validationLabels = validationSet(size(validationSet,1), :);

trainingFeatures = trainingSet(1:size(trainingSet,1)-1, :);
validationFeatures = validationSet(1:size(validationSet,1)-1, :);

size(trainingFeatures')
size(validationFeatures')

save(filename, 'trainingFeatures', 'trainingLabels', 'validationFeatures', 'validationLabels', '-v7.3');
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