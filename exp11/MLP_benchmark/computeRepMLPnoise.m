function F = computeRepMLPnoise(Q, model, param)

% param.stride is stride size between each window
% param.windowSize is the number of frames per window

model = model.weights;
weight = model.W1;
bias = model.b1;

Q = preprocessQMLPnoise(Q, param.PFLAG);

curRep = [];
for i=1:param.stride:size(Q,2)-param.windowSize+1
    window = reshape(Q(:,i:i+param.windowSize-1),param.totalbins*param.windowSize,1);
    curRep = cat(2,curRep,(window'*weight + bias)');
end

med = median(curRep,2) * ones(1,size(curRep,2));
   
F = (curRep > med);

end