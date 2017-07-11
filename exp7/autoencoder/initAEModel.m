function initAEModel( saveFilename, parameter )

%% load model
model = load(saveFilename);
W_list = model.W_list;
b_list = model.b_list;
num_layer = model.num_layer;

%% param config
parameter.m = 20;
parameter.medianThreshold = 1;
parameter.useDelta = 0;
parameter.hop = 5;

%% Save to file
disp(['Saving AE models to file']);
save(saveFilename,'W_list','b_list','parameter', 'num_layer');

end

