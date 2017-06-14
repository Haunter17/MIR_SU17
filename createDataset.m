function [trainingSet,testingSet] = createDataset (Q, label, strideSize, windowSize)

% With downsampling rate = 15, 169 coefficients/frame, we have approximately 16 frames/sec
% Training duration: 10 secs = 160 frames
% Testing duration: 5 secs = 80 frames
trainingVec = [];
testingVec = [];
trainingDuration = 160;
testingDuration = 80;

for col=1:trainingDuration+testingDuration: size(Q,2)
    for i=col:strideSize:min(col+trainingDuration-windowSize, size(Q,2)-windowSize+1)
        trainingVec = cat(2,trainingVec,reshape(Q(:,i:i+windowSize-1)',169*windowSize,1));
    end
    if col + trainingDuration - 1 >= size(Q, 2)
    	break
    end
    for i=col+trainingDuration+1:strideSize:min(col+trainingDuration+testingDuration-windowSize,size(Q,2)-windowSize+1)
        testingVec = cat(2,testingVec,reshape(Q(:,i:i+windowSize-1)',169*windowSize,1));
    end
end

% Volume normalization
trainingVec = normc(trainingVec);
testingVec = normc(testingVec);

% Add labels
trainingLabel = ones(size(trainingVec',1), 1) * label;
testingLabel = ones(size(testingVec',1), 1) * label;
trainingSet = [trainingVec' trainingLabel];
testingSet = [testingVec' testingLabel];
