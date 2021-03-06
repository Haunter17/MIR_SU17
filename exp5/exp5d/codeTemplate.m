addpath('cqt/');
%% Task 1: run live song id system
% precompute CQT on filelist
artist = 'taylorswift';
filelist = strcat('audio/', artist, '_ref.list');
outdir = strcat(artist, '_out/');
mkdir(outdir)

computeQSpecBatch(filelist,outdir);
%% Load data into a huge cell array
param.precomputeCQT = 1;
param.precomputeCQTdir = outdir;
downsamplingRate = 15;
createDatasetBatch(filelist, strcat('exp5d_d15_1s.mat'), downsamplingRate, outdir);