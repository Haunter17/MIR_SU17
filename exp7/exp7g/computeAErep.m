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
if isfield(parameter,'deltaDelay')==0
    parameter.deltaDelay=16;
end

F = zeros(parameter.numFeatures, ceil((size(spec, 2) - parameter.m) / ...
    parameter.hop));
for col = 1 : parameter.hop : size(spec, 2) - parameter.m
    X = spec(:, col : col + parameter.m - 1);
    X = X(:)';
    feature = poslin(X * W + b);        
    F(:, ceil(col / parameter.hop)) = feature;    
end
F = F > 0;
end