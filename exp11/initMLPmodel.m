function initMLPmodel(weightFile, modelFile)
    % initialize all parameters
    % save model and parameters together into model.mat file

    % CQT parameters
    parameter = [];
    parameter.targetsr = 22050;
    parameter.B = 24;
    parameter.fmin = 130.81;
    parameter.fmax = 4186.01;
    parameter.gamma = 0;
    parameter.precomputeCQT = 0;
    parameter.totalbins = 121;

    % MLP parameters
    parameter.PFLAG = 1; % Log transformation
    parameter.stride = 1;
    parameter.windowSize = 32;
    parameter.numFeatures = 64;
    parameter.hop = 8;
    parameter.queryLen = 6; % in seconds

    % load MLP weight
    weights = load(weightFile);

    % save model with parameters
    save(modelFile, 'weights', 'parameter')
end