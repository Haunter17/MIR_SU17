addpath('../../cqt/');
addpath('../exp7d/');
REPFLAG = 1; % 1 for baseline system
repNameList = {'hashprint', 'randomized', 'AE'};
%% Parallel computing setup
curPool = gcp('nocreate'); 
if (isempty(curPool))
    myCluster = parcluster('local');
    numWorkers = myCluster.NumWorkers;
    % create a parallel pool with the number of workers in the cluster`
    pool = parpool(ceil(numWorkers * 0.75));
end

%% precompute CQT on reflist
artist = 'taylorswift';
reflist = strcat(artist, '_ref.list');
noisylist = strcat(artist, '_noisy.list');
outdir = strcat(artist, '_out/');
mkdir(outdir)

noisyNameList = {'original', '90% speed', '95% speed', '105% speed', '110% speed',...
    'Amplitude scaling -15dB', 'Amplitude scaling -10dB', 'Amplitude scaling -5dB', ...
    'Amplitude scaling 5dB', 'Amplitude scaling 10dB', 'Amplitude scaling 15dB', ...
    'Pitch shift -1', 'Pitch shift -0.5', 'Pitch shift 0.5', 'Pitch shift 1', ...
    'Reverberation: Drinkward', 'Reverberation: Galileo', 'Reverberation: Shanahan Front', ...
    'Reverberation: Shanahan Middle', 'Reverberation: Matlab Generated', ...
    'Crowd: SNR = -15dB', 'Crowd: SNR = -10dB', 'Crowd: SNR = -5dB', 'Crowd: SNR = 0dB', ...
    'Crowd: SNR = 5dB', 'Crowd: SNR = 10dB', 'Crowd: SNR = 15dB', 'Crowd: SNR = 100dB', ...
    'Restaurant: SNR = -15dB', 'Restaurant: SNR = -10dB', 'Restaurant: SNR = -5dB', ...
    'Restaurant: SNR = 0dB', 'Restaurant: SNR = 5dB', 'Restaurant: SNR = 10dB', ...
    'Restaurant: SNR = 15dB', 'Restaurant: SNR = 100dB', ...
    'AWGN: SNR = -15dB', 'AWGN: SNR = -10dB', 'AWGN: SNR = -5dB', 'AWGN: SNR = 0dB', ...
    'AWGN: SNR = 5dB', 'AWGN: SNR = 10dB', 'AWGN: SNR = 15dB', 'AWGN: SNR = 100dB'
    };

rateList = ones(1, length(noisyNameList));
rateList(2) = 0.9;
rateList(3) = 0.95;
rateList(4) = 1.05;
rateList(5) = 1.1;

param.precomputeCQT = 1;
param.precomputeCQTdir = outdir;
computeQSpecBatch(reflist,outdir);
computeQSpecBatch(noisylist, outdir);

%% learn models and generate representations
param.m = -1;
modelFile = strcat(outdir, 'model.mat');

% switch for different representations
switch REPFLAG
    case 1
        param.m = 20;
        learnHashprintModel(reflist, modelFile, param);
        representations = getHashprintRepresentation(modelFile, noisylist);
    case 2
        param.numFiltersList = [256];
        RandomProjectionModelInit(reflist, modelFile, param);
        representations = RandomProjectionModelGetRep(modelFile, 0, noisylist);
    case 3
        param.numFeatures = 64;
        param.m = 20;
        representations = getAErep(modelFile, noisylist);
    otherwise
        pass
end

%% evaluate representations
[pctList, corrList] = evaluateRepresentation(representations, noisyNameList, rateList);
reportfile = strcat(outdir, repNameList{REPFLAG}, '-', num2str(param.m), datestr(now, '_HH-MM-SS-FFF'), '.out');
generateEvalReport(noisyNameList, pctList, corrList, reportfile);
