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
testSet = [];
label = 0;

while ischar(curfile)
	disp(['-- Generating data matrix on file ', int2str(label)]);
    [pathstr,name,ext] = fileparts(curfile);
    cqt_file = strcat(outdir,name,'.mat');
    Qfile = load(cqt_file); % loads Q (struct)
    QMat = abs(Qfile.Q.c);
    
    [trainVec, testVec] = createDataset(QMat, label);
    trainingSet = cat(1, trainingSet, trainVec);
    testSet = cat(1, testSet, testVec);
    
    curfile = fgetl(fid);
    label = label + 1;
end

% Note: Downsampling is done before shuffling data to ensure that we sample
% from every track.
disp(['==> Downsampling Data'])
trainingSet = trainingSet(1:downsamplingRate:size(trainingSet,1),:);
testSet = testSet(1:downsamplingRate:size(testSet,1),:);

disp(['==> Shuffling data...'])
trainingSet = trainingSet(randperm(size(trainingSet, 1)), :)';
testSet = testSet(randperm(size(testSet, 1)), :)';

disp(['==> Separate coefficients and labels'])
trainingLabels = trainingSet(size(trainingSet,1), :);
validationLabels = testSet(size(testSet,1), :);
trainingFeatures = trainingSet(1:size(trainingSet,1)-1, :);
validationFeatures = testSet(1:size(testSet,1)-1, :);

save(filename, 'trainingFeatures', 'validationFeatures', 'trainingLabels', 'validationLabels', '-v7.3');