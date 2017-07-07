addpath('../');

% quick test to make sure the system works
% param.numFiltersList = [2];
% param.numFilterRows = [121];
% param.numFilterCols = [3];
% param.ThresholdStrategy = 'zero';
% param.nonlinearity = 'relu';
% param.DeltaFeatures = false
% param.nameForLegend = 'DONT USE THIS - SHORT TEST';
% exp7dRunner

param.FilterType = 'RowMean';
param.downsamplingRate = 3;
param.DeltaDelay = -1;

% run with different numbers of context frames - first mimic the baseline with 20
param.expNum = 2;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [20];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = '20 frames';
exp7dRunner

% now run with 1, 10, 30, and 40 context frames

% 1
param.expNum = 2;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [1];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = '1 frame';
param.DownsamplingRate = 3;
exp7dRunner

% 10
param.expNum = 2;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [10];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = '10 frames';
param.DownsamplingRate = 3;
exp7dRunner

% 30
param.expNum = 2;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [30];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = '30 frames';
param.DownsamplingRate = 3;
exp7dRunner

% 40
param.expNum = 2;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [40];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = '40 frames';
param.DownsamplingRate = 3;
exp7dRunner

% 80
param.expNum = 2;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [80];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = '80 frames';
param.DownsamplingRate = 3;
exp7dRunner


% 160
param.expNum = 2;
param.numFiltersList = [64];
param.numFilterRows = [121];
param.numFilterCols = [160];
param.ThresholdStrategy = 'zero';
param.nonlinearity = 'relu';
param.DeltaFeatures = false;
param.nameForLegend = '160 frames';
param.DownsamplingRate = 3;
exp7dRunner