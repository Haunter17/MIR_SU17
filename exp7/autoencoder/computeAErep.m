function F = computeAErep(Q, model, parameter)

if nargin < 3
    parameter=[];
end

W_list = model.W_list;
b_list = model.b_list;
num_layer = model.num_layer;

if isfield(model, 'parameter')
    parameter = model.parameter;
end

if model.num_layer == 1
    W_list = reshape(W_list, size(W_list, 2), size(W_list, 3));
    W_list = mat2cell(W_list, size(W_list, 1));
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
    parameter.numFeatures=size(b_list,2);
end
if isfield(parameter, 'useDelta')==0
    parameter.useDelta = 0;
end
if isfield(parameter,'deltaDelay')==0
    parameter.deltaDelay=16;
end
if isfield(parameter, 'medianThreshold')==0
    parameter.medianThreshold = 0;
end

b_list = mat2cell(b_list, ones(num_layer, 1), parameter.numFeatures);

spec = preprocessQspec(Q);
features = zeros(parameter.numFeatures, ceil((size(spec, 2) - parameter.m) / ...
    parameter.hop));

for col = 1 : parameter.hop : size(spec, 2) - parameter.m
    X = spec(:, col : col + parameter.m - 1);
    X = X(:)';
    a = X;
    for layer = 1 : num_layer - 1
        W = W_list{layer};
        b = b_list{layer};
        a = poslin(a * W + b);
    end
    % do not take non-linearity at the last layer
    W = W_list{num_layer};
    b = b_list{num_layer};
    f = a * W + b; 
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
