function downsampleChannels()

filename = '(separate features & labels) exp1a_smallDataset_71_7.mat';
saveFilename = 'exp1m_taylorswift_12.mat';
data = load(filename);

testM = data.validationFeatures;
trainM = data.trainingFeatures;
trainingLabels = data.trainingLabels;
validationLabels = data.validationLabels;

numSamples = size(trainM,2);
trainingFeatures = zeros(61,numSamples);

for col = 1:numSamples
    for row = 1:2:120
        trainingFeatures((row+1)/2, col) = (trainM(row,col)+trainM(row+1,col))/2;
    end
    trainingFeatures(61,col) = trainM(121,col);
end

disp(size(trainingFeatures));

numSamples = size(testM,2);
validationFeatures = zeros(61,numSamples);

for col = 1:numSamples
    for row = 1:2:120
        validationFeatures((row+1)/2, col) = (testM(row,col)+testM(row+1,col))/2;
    end
    validationFeatures(61,col) = testM(121,col);
end

save(saveFilename, 'trainingFeatures', 'trainingLabels', 'validationFeatures', 'validationLabels', '-v7.3');
