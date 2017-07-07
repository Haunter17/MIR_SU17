addpath('../');


param.FilterType = 'RowMean';
param.DeltaDelay = -1;

% try both zero and median (across row) thresholding
param.expNum = 0;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [20];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = 'zero thresholding';
param.DownsamplingRate = 3;
exp7dRunner

% run with different numbers of context frames - first mimic the baseline with 20
param.expNum = 0;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [20];
param.ThresholdStrategy = 'median';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = 'median thresholding';
param.DownsamplingRate = 3;
exp7dRunner