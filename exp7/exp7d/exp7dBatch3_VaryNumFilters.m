addpath('../');

% run with different numbers of filters

param.DownsamplingRate = 3;
param.FilterType = 'RowMean';
param.numFilterRows = [121];
param.numFilterCols = [20];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.expNum = 3;
param.DeltaDelay = -1;


% 32
param.numFiltersList = [32];
param.nameForLegend = '32 filters';
exp7dRunner

% 64
param.numFiltersList = [64];
param.nameForLegend = '64 filters';
exp7dRunner

% 128
param.numFiltersList = [128];
param.nameForLegend = '128 filters';
exp7dRunner

% 256
param.numFiltersList = [256];
param.nameForLegend = '256 filters';
exp7dRunner

% 512
param.numFiltersList = [512];
param.nameForLegend = '512 filters';
exp7dRunner

% 1024
param.numFiltersList = [1024];
param.nameForLegend = '1024 filters';
exp7dRunner