function initAEModel( saveFilename, parameter )

%% load model
load(saveFilename);

%% Save to file
disp(['Saving AE models to file']);
save(saveFilename,'W','b','parameter');

end

