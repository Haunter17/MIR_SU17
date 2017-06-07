addpath('../../cqt/');
%% Parallel computing setup
curPool = gcp('nocreate'); 
if (isempty(curPool))
    myCluster = parcluster('local');
    numWorkers = myCluster.NumWorkers;
    % create a parallel pool with the number of workers in the cluster`
    pool = parpool(ceil(numWorkers * 0.75));
end

%% Task 1: run live song id system
%% precompute CQT on filelist
artist = 'taylorswift';
filelist = strcat('../../audio/', artist, '_ref.list');
outdir = strcat('../../', artist, '_out/');
mkdir(outdir)


% change subsampling rate in computeQSpecBatch and runPCAQueries
computeQSpecBatch(filelist,outdir);
%% Load data into a huge cell array
param.precomputeCQT = 1;
param.precomputeCQTdir = outdir;
downsamplingRate = 15;

createDatasetBatch(filelist, strcat(outdir,'data_s15.mat'), downsamplingRate, outdir)
