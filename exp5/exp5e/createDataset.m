function [trainingSet, validationSet, window_trainingSet, window_validationSet] = createDataset (Q, label, strideSize, windowSize)

% With downsampling rate = 7 (not smoothing), 169 coefficients/frame, we have approximately 34 frames/sec
% Training duration: 10 secs = 340 frames
% Testing duration: 5 secs = 170 frames
trainingVec = [];
validationVec = [];
window_trainingVec = [];
window_validationVec = [];
trainingDuration = windowSize * 10;
testingDuration = windowSize * 5;

% Volume normalization
Q = normc(Q);

for col=1:trainingDuration+testingDuration: size(Q,2)
    % Create training dataset
    for i=col:strideSize:min(col+trainingDuration-windowSize, size(Q,2)-windowSize+1)
        trainingVec = cat(2, trainingVec, Q(:,i:i+windowSize-1)');
    end
    
    for i=col:strideSize:min(col+trainingDuration-windowSize, size(Q,2)-windowSize+1)
        window_trainingVec = cat(2,window_trainingVec,reshape(Q(:,i:i+windowSize-1)',169*windowSize,1));
    end
    
    if col + trainingDuration - 1 >= size(Q, 2)
    	break
    end
    
    % Create validation dataset
    for i=col+trainingDuration+1:strideSize:min(col+trainingDuration+testingDuration-windowSize,size(Q,2)-windowSize+1)
        validationVec = cat(2,validationVec,Q(:,i:i+windowSize-1)');
    end
    
    for i=col+trainingDuration+1:strideSize:min(col+trainingDuration+testingDuration-windowSize,size(Q,2)-windowSize+1)
        window_validationVec = cat(2,window_validationVec,reshape(Q(:,i:i+windowSize-1)',169*windowSize,1));
    end
end

disp(size(trainingVec));
disp(size(validationVec));
disp(size(window_trainingVec));
disp(size(window_validationVec));

% Add labels
trainingLabel = ones(size(trainingVec',1), 1) * label;
validationLabel = ones(size(validationVec',1), 1) * label;
window_trainingLabel = ones(size(window_trainingVec',1), 1) * label;
window_validationLabel = ones(size(window_validationVec',1), 1) * label;

trainingSet = [trainingVec' trainingLabel];
validationSet = [validationVec' validationLabel];
window_trainingSet = [window_trainingVec' window_trainingLabel];
window_validationSet = [window_validationVec' window_validationLabel];