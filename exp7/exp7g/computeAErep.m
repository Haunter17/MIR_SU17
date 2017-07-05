function F = computeAErep(spec, W, b, parameter)

if nargin < 4
    parameter=[];
end
if isfield(parameter,'m')==0
    parameter.m=20;
end
if isfield(parameter,'tao')==0
    parameter.tao=1;
end
if isfield(parameter,'hop')==0
    parameter.hop=5;
end
if isfield(parameter,'numFeatures')==0
    parameter.numFeatures=size(W,2);
end
if isfield(parameter, 'useDelta')==0
    parameter.delta = 0;
end
if isfield(parameter,'deltaDelay')==0
    parameter.deltaDelay=16;
end
if isfield(parameter, 'medianThreshold')==0
    parameter.threshold = 0;
end

features = zeros(parameter.numFeatures, ceil((size(spec, 2) - parameter.m) / ...
    parameter.hop));
for col = 1 : parameter.hop : size(spec, 2) - parameter.m
    X = spec(:, col : col + parameter.m - 1);
    X = X(:)';
    f = X * W + b;
    features(:, ceil(col / parameter.hop)) = f;    
end

F = features;
if parameter.useDelta
    deltas = features(:,1:(size(features,2)-parameter.deltaDelay)) - features(:,(1+parameter.deltaDelay):end);
    F = deltas;
end
    
if parameter.medianThreshold
    F = F > repmat(median(F, 2), 1, size(F, 2));
else
    F = F > 0;
end
end
