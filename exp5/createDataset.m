function [trainingSet, validationSet, window_trainingSet, window_testingSet] = createDataset (Q, label, strideSize, windowSize)

trainingVec = [];
validationVec = [];
window_trainingVec = [];
window_testingVec = [];
trainingDuration = windowSize * 10;
testingDuration = windowSize * 5;

% Volume normalization
Q = normc(Q);

for col=1:trainingDuration+testingDuration: size(Q,2)
    trainingVec = cat(2, trainingVec, Q(:, col : min(col + trainingDuration - 1, size(Q, 2))));
    for i=col:strideSize:min(col+trainingDuration-windowSize, size(Q,2)-windowSize+1)
        window_trainingVec = cat(2,window_trainingVec,reshape(Q(:,i:i+windowSize-1)',169*windowSize,1));
    end
    
    if col + trainingDuration - 1 >= size(Q, 2)
    	break
    end
    
    validationVec = cat(2, validationVec, ...
    	Q(:, col + trainingDuration : min(col + trainingDuration + testingDuration - 1, size(Q, 2))));
    for i=col+trainingDuration+1:strideSize:min(col+trainingDuration+testingDuration-windowSize,size(Q,2)-windowSize+1)
        %disp(size(Q(:,i:i+windowSize-1)));
        window_testingVec = cat(2,window_testingVec,reshape(Q(:,i:i+windowSize-1)',169*windowSize,1));
    end
end

% Add labels
trainingLabel = ones(size(trainingVec',1), 1) * label;
validationLabel = ones(size(validationVec',1), 1) * label;
window_trainingLabel = ones(size(window_trainingVec',1), 1) * label;
window_testingLabel = ones(size(window_testingVec',1), 1) * label;

trainingSet = [trainingVec' trainingLabel];
validationSet = [validationVec' validationLabel];
window_trainingSet = [window_trainingVec' window_trainingLabel];
window_testingSet = [window_testingVec' window_testingLabel];