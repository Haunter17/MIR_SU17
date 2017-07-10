function initAEModel( saveFilename, parameter )

%% load model
model = load(saveFilename);
W = model.W;
b = model.b;
parameter = model.parameter;

%% Save to file
disp(['Saving AE models to file']);
save(saveFilename,'W','b','parameter');

end

