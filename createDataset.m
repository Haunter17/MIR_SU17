function [trainingSet,testingSet] = createDataset (Q, label)

% 250 frames/sec
% Training duration: 12 secs = 966 frames = 2 col
% Testing duration: 6 secs = 483 frames = 1 col
trainingVec = [];
testingVec = [];
trainingDuration = 2;
testingDuration = 1;

for col=1:trainingDuration+testingDuration: size(Q,2)
	if col + trainingDuration + testingDuration - 1 >= size(Q, 2)
    	break
    end
    trainingVec = cat(1, trainingVec, Q(:, col : col + trainingDuration - 1)');
    testingVec = cat(1, testingVec, ...
    	Q(:, col + trainingDuration : trainingDuration + testingDuration - 1)');
end

trainingLabel = ones(size(trainingVec,1), 1) * label;
testingLabel = ones(size(testingVec,1), 1) * label;
trainingSet = [trainingVec trainingLabel];
testingSet = [testingVec testingLabel];
