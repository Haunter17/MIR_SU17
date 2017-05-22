function [trainingSet,testingSet] = createDataset (Q, label)

% 250 frames/sec
% Training duration: 10 secs = 2500 frames
% Testing duration: 5 secs = 1250 frames
trainingVec = [];
testingVec = [];
trainingDuration = 2500;
testingDuration = 1250;

for col=1:size(Q,2):trainingDuration+testingDuration
    trainingVec = cat(1, trainingVec, Q(:, col:col+trainingDuration-1)');
    testingVec = cat(1, testingVec, Q(:, col+trainingDuration:col+trainingDuration+testingDuration-1)');
end

trainingLabel = ones(size(trainingVec,1), 1) * label;
testingLabel = ones(size(testingVec,1), 1) * label;
trainingSet = [trainingVec trainingLabel];
testingSet = [testingVec testingLabel];

