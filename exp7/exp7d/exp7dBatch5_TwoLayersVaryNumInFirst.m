addpath('../');

% Keep the output dimension at 64, vary the number of layers it passes through to get there
% this means the last numFilters will be 64, and we'll vary the previous one

% common parameters
param.expNum = 5;
param.numFilterRows = [121];
param.numFilterCols = [20];
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.DownsamplingRate = 3;
param.ThresholdStrategy = 'zero';
param.FilterType = 'FullMean';
param.DeltaDelay = -1;

% 1
param.numFiltersList = [2, 64];
param.nameForLegend = '2 filters in first';
exp7dRunner

% 8
param.numFiltersList = [8, 64];
param.nameForLegend = '8 filters in first';
exp7dRunner

% 32
param.numFiltersList = [32, 64];
param.nameForLegend = '32 filters in first';
exp7dRunner

% 64
param.numFiltersList = [64, 64];
param.nameForLegend = '64 filters in first';
exp7dRunner

% 128
param.numFiltersList = [128, 64];
param.nameForLegend = '128 filters in first';
exp7dRunner

% 256
param.numFiltersList = [256, 64];
param.nameForLegend = '256 filters in first';
exp7dRunner

% 512
param.numFiltersList = [512, 64];
param.nameForLegend = '512 filters in first';
exp7dRunner



