addpath('../');

% try filters where each row has a mean of 0, versus where 
% the full filter is set to have a mean of 0

% common parameters
param.expNum = 6;
param.numFilterRows = [121];
param.numFilterCols = [20];
param.numFiltersList = [64];
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.DownsamplingRate = 3;
param.ThresholdStrategy = 'zero';
param.DeltaDelay = -1;

% row mean
param.FilterType = 'RowMean';
param.nameForLegend = 'Row Mean Filters';
exp7dRunner

% full mean
param.FilterType = 'FullMean';
param.nameForLegend = 'Full Mean Filters';
exp7dRunner