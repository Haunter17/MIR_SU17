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
dsr_list = [5, 10, 20, 30];

for i = 1:4
    downsamplingRate = dsr_list(i);
    createDatasetBatch(filelist, strcat('exp5c_d',int2str(downsamplingRate),'.mat'), downsamplingRate, outdir)
end