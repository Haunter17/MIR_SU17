function initAEModel( saveFilename, parameter )

%% load model
model = load(saveFilename);
W = model.W;
b = model.b;

%% param config
parameter.m = 20;
parameter.medianThreshold = 1;
parameter.useDelta = 0;
parameter.hop = 5;

%% Save to file
disp(['Saving AE models to file']);
save(saveFilename,'W','b','parameter');

end

