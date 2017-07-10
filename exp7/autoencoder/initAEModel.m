function initAEModel( saveFilename, parameter )

%% load model
model = load(saveFilename);
W = model.W;
b = model.b;
parameter = model.parameter;

%% param config
param.m = 20;
param.medianThreshold = 1;
param.useDelta = 0;
param.hop = 5;

%% Save to file
disp(['Saving AE models to file']);
save(saveFilename,'W','b','parameter');

end

