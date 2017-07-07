addpath('../');


% try plain window features vs. delta features (looking at how features are changing
%	to get your binary print rather than looking at the value of the feature alone)
param.FilterType = 'RowMean';
param.expNum = 1;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [20];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DownsamplingRate = 3;

% window features
param.DeltaFeatures = false;
param.DeltaDelay = -1;
param.nameForLegend = 'window features';
exp7dRunner

% delta features, D = 8
param.DeltaFeatures = true;
param.DeltaDelay = 8;
param.nameForLegend = 'Delta features, D = 8';
exp7dRunner

% delta features, D = 16
param.DeltaFeatures = true;
param.DeltaDelay = 16;
param.nameForLegend = 'Delta features, D = 16';
exp7dRunner

% delta features, D = 32
param.DeltaFeatures = true;
param.DeltaDelay = 32;
param.nameForLegend = 'Delta features, D = 32';
exp7dRunner

% delta features, D = 64
param.DeltaFeatures = true;
param.DeltaDelay = 64;
param.nameForLegend = 'Delta features, D = 64';
exp7dRunner
