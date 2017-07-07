addpath('../');

% parameter settings that stay constant
param.expNum = 4;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [20];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.FilterType = 'FullMean';
param.DeltaDelay = -1;

% run with different downsampling rates

% 1
param.nameForLegend = 'Downsampling of 1';
param.DownsamplingRate = 1;
exp7dRunner

% 3
param.nameForLegend = 'Downsampling of 3';
param.DownsamplingRate = 3;
exp7dRunner

% 15
param.nameForLegend = 'Downsampling of 15';
param.DownsamplingRate = 15;
exp7dRunner

% 30
param.nameForLegend = 'Downsampling of 30';
param.DownsamplingRate = 30;
exp7dRunner

% 45
param.nameForLegend = 'Downsampling of 45';
param.DownsamplingRate = 45;
exp7dRunner