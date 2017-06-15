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
window_trainSet = [];
window_testSet = [];
label = 0;

while ischar(curfile)
	disp(['-- Generating data matrix on file ', int2str(label)]);
    [pathstr,name,ext] = fileparts(curfile);
    cqt_file = strcat(outdir,name,'.mat');
    Qfile = load(cqt_file); % loads Q (struct)
    % preprocessing of Q: cubic root
    Q_Mat = nthroot(abs(Qfile.Q.c), downsamplingRate);
    % average every n columns
    QMat = Q_Mat(:,1:downsamplingRate:size(Q_Mat,2));

    num_frame_agg = floor(242/downsamplingRate);
    strideSize = floor(num_frame_agg/4);
    
    [trainVec, validationVec, window_trainVec, window_testVec] = createDataset(QMat, label, strideSize, num_frame_agg);
    trainingSet = cat(1, trainingSet, trainVec);
    validationSet = cat(1, validationSet, validationVec);
    window_trainSet = cat(1, window_trainSet, window_trainVec);
    window_testSet = cat(1, window_testSet, window_testVec);
    
    curfile = fgetl(fid);
    label = label + 1;
end
 
disp(['==> Shuffling data...'])
trainingSet = trainingSet(randperm(size(trainingSet, 1)), :)';
validationSet =validationSet(randperm(size(validationSet, 1)), :)';
window_trainSet = window_trainSet(randperm(size(window_trainSet, 1)), :)';
window_testSet = window_testSet(randperm(size(window_testSet, 1)), :)';

%disp(['==> Slice data...'])
%trainingSet = trainingSet(:,1:min(size(trainingSet,2),100000));
%validationSet = validationSet(:,1:min(size(validationSet,2),50000));
%testSet = testSet(:,1:min(size(testSet,2),19000));

disp(['==> Separate coefficients and labels'])
trainingLabels = trainingSet(size(trainingSet,1), :);
validationLabels = validationSet(size(validationSet,1), :);
window_trainLabels = window_trainSet(size(window_trainSet,1), :);
window_testLabels = window_testSet(size(window_testSet,1), :);

trainingFeatures = trainingSet(1:size(trainingSet,1)-1, :);
validationFeatures = validationSet(1:size(validationSet,1)-1, :);
window_trainFeatures = window_trainSet(1:size(window_trainSet,1)-1, :);
window_testFeatures = window_testSet(1:size(window_testSet,1)-1, :);

size(trainingFeatures')
size(validationFeatures')
size(window_trainFeatures')
size(window_testFeatures')

save(filename, 'trainingFeatures', 'trainingLabels', 'validationFeatures', 'validationLabels', ...
    'window_trainFeatures', 'window_trainLabels', 'window_testFeatures', 'window_testLabels', '-v7.3');
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
