addpath('../../cqt/');
addpath('../exp7/benchmark/');

%% Parallel computing setup
curPool = gcp('nocreate'); 
if (isempty(curPool))
    myCluster = parcluster('local');
    numWorkers = myCluster.NumWorkers;
    % create a parallel pool with the number of workers in the cluster`
    pool = parpool(ceil(numWorkers * 0.75));
end

%% param setup
if isfield(parameter,'m')==0
    parameter.m=20;
end
if isfield(parameter,'tao')==0
    parameter.tao=1;
end
if isfield(parameter,'hop')==0
    parameter.hop=5;
end
if isfield(parameter,'numFeatures')==0
    parameter.numFeatures=64;
end
if isfield(parameter,'deltaDelay')==0
    parameter.deltaDelay=16;
end
if isfield(parameter,'precomputeCQT')==0
    parameter.precomputeCQT = 0;
end

parameter.downsample = 12;
parameter.m = floor(483 / (parameter.downsample / 3));

%% config
artist = 'taylorswift';
prompt = 'Please enter the name of the artist.\n';
artist = input(prompt, 's');
reflist = strcat('../exp7/benchmark/audio/', artist, '_ref.list');
vallist = strcat('../exp7/benchmark/audio/', artist, '_fullval.list');
outdir = strcat('../exp7/benchmark/', artist, '_out/');
mkdir(outdir)
parameter.precomputeCQTdir = outdir;
savename = strcat(outdir, artist, '_data.mat');

%% training data
DTrain = CNNDataHelper(reflist, parameter);

%% validation data
valLabelFile = strcat('../exp7/benchmark/audio/', ...
    artist, '_fullvaltoref.csv');
valLabels = csvread(valLabelFile);
DVal = CNNDataHelper(vallist, parameter, valLabels);
save(savename, 'DTrain', 'Dval', '-v7.3');
