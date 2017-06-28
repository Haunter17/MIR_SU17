addpath('cqt/');
%% Task 1: run live song id system
% precompute CQT on filelist
artist = 'taylorswift';
filelist = strcat('audio/taylorswift_ref.list');
outdir = strcat(artist, '_out/');
mkdir(outdir)

computeQSpecBatch(filelist,outdir);
%% Load data into a huge cell array
param.precomputeCQT = 1;
param.precomputeCQTdir = outdir;
downsamplingRate = 7;
createDatasetBatch(filelist, strcat('exp7f_5g_cbrt.mat'), downsamplingRate, outdir);
