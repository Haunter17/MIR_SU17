
addpath('../');

% try different layer numbers with a constant number of filters in each layer
param.expNum = 8;
param.numFilterRows = [121];
param.numFilterCols = [20];
param.nonlinearity = 'sigmoid';
param.DeltaFeatures = false;
param.DownsamplingRate = 3;
param.ThresholdStrategy = 'zero';
param.FilterType = 'FullMean';
param.DeltaDelay = -1;


% first try with a constant number of filters in each layer
param.numFiltersList = [64];
param.nameForLegend = 'Single Layer with 64';
exp7dRunner

param.numFiltersList = [64, 64];
param.nameForLegend = 'Two Layers with 64';
exp7dRunner

param.numFiltersList = [64, 64, 64];
param.nameForLegend = 'Three Layers with 64';
exp7dRunner

param.numFiltersList = [64, 64, 64, 64];
param.nameForLegend = 'Four Layers with 64';
exp7dRunner

param.numFiltersList = [64, 64, 64, 64, 64];
param.nameForLegend = 'Five Layers with 64';
exp7dRunner

% then try linearly decreasing from 128 filters to 64 filters as the layers go on
param.numFiltersList = [128, 64];
param.nameForLegend = 'Two layers: 128, 64';
exp7dRunner

param.numFiltersList = [128, 96, 64];
param.nameForLegend = 'Three layers: 128, 96, 64';
exp7dRunner

param.numFiltersList = [128, 107, 86, 64];
param.nameForLegend = 'Four layers: 128, 107, 86, 64';
exp7dRunner


