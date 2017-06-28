function [trainingSet, validationSet] = createDataset (Q, label, strideSize, windowSize)

trainingVec = [];
validationVec = [];
trainingDuration = windowSize * 10;
testingDuration = windowSize * 5;

% Volume normalization
Q = normc(Q);

for col=1:trainingDuration+testingDuration: size(Q,2)
    % Create training dataset
    for i=col:strideSize:min(col+trainingDuration-windowSize, size(Q,2)-windowSize+1)
        trainingVec = cat(2, trainingVec, Q(:,i:i+windowSize-1)');
    end
    
    if col + trainingDuration - 1 >= size(Q, 2)
    	break
    end
    
    % Create validation dataset
    for i=col+trainingDuration+1:strideSize:min(col+trainingDuration+testingDuration-windowSize,size(Q,2)-windowSize+1)
        validationVec = cat(2,validationVec,Q(:,i:i+windowSize-1)');
    end
end

disp(size(trainingVec));
disp(size(validationVec));

% Add labels
trainingLabel = ones(size(trainingVec',1), 1) * label;
validationLabel = ones(size(validationVec',1), 1) * label;

trainingSet = [trainingVec' trainingLabel];
validationSet = [validationVec' validationLabel];