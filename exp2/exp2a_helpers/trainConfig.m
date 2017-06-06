function trainConfig(saveFilename, verticalSize, horizontalSize, numSampPerRef)
%
%   Set up the config for the autoencoder and save it in a file
%   Notice: the configs in perfwb and dperf_wb still have to be hard coded

N = 16; % number of 121 x 1 vectors
M = 16; % number of 1 x 30 vectors
disp(['Saving autoencoder configs to file']);
save(saveFilename, 'verticalSize', 'horizontalSize', 'numSampPerRef', 'N', 'M');

end
