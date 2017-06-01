function [trainingSet,testingSet] = createDataset (Q, label)

% 250 frames/sec
% Training duration: 10 secs = 966 frames = 2 col
% Testing duration: 5 secs = 483 frames = 1 col
trainingVec = [];
testingVec = [];
trainingDuration = 160;
testingDuration = 80;

for col= 1:trainingDuration+testingDuration: size(Q,2) - (trainingDuration + testingDuration) + 1
    trainingVec = cat(1, trainingVec, Q(:, col : col + trainingDuration - 1)');
    testingVec = cat(1, testingVec, ...
    	Q(:, col + trainingDuration : col + trainingDuration + testingDuration - 1)');
end

trainingLabel = ones(size(trainingVec,1), 1) * label;
testingLabel = ones(size(testingVec,1), 1) * label;
trainingSet = [trainingVec trainingLabel];
testingSet = [testingVec testingLabel];
