% code to move over from our old file format to our new file format
filename = 'exp1/chromeo_smallDataset_44_7_old.mat';
saveFilename = 'exp1/chromeo_smallDataset_44_7.mat';
data = load(filename);

testSet = data.testSet;
trainingSet = data.trainingSet;
% grab the labels
validationLabels = testSet(end, :);
trainingLabels = trainingSet(end, :);

% grab the features
validationFeatures = testSet(1:end-1,:);
trainingFeatures = trainingSet(1:end-1,:);

save(saveFilename, 'trainingFeatures', 'trainingLabels', 'validationFeatures', 'validationLabels', '-v7.3');