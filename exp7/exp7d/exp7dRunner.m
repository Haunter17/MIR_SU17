addpath('../../cqt/');
REPFLAG = 2; % 1 for baseline system
repNameList = {'hashprint', 'randomized'};
%% Parallel computing setup
curPool = gcp('nocreate'); 
% if (isempty(curPool))
%     myCluster = parcluster('local');
%     numWorkers = myCluster.NumWorkers;
%     % create a parallel pool with the number of workers in the cluster`
%     pool = parpool(ceil(numWorkers * 0.75));
% end

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
        % param.numFiltersList = [256];
        % param.numFilterRows = [121];
        % param.numFilterCols = [121];
        experimentIdentifierString = sprintf('exp7d_%g_%gx%g_%sFilts_%s_%sThreshold_DeltaFeatures=%d_DeltaDelay=%g_Downsampling=%g_FilterType=%s', param.expNum, param.numFilterRows, param.numFilterCols, mat2str(param.numFiltersList), param.nonlinearity, param.ThresholdStrategy, param.DeltaFeatures, param.DeltaDelay, param.DownsamplingRate, param.FilterType);
        fprintf('Starting %s\n', experimentIdentifierString)
        RandomProjectionModelInit(reflist, modelFile, param);
        representations = RandomProjectionModelGetRep(modelFile, 0, noisylist);
    case 3
        param.numFeatures = 64;
        param.m = 20;
        param.medianThreshold = 0;
        param.useDelta = 0;
        representations = getAERepresentation(modelFile, noisylist, param);
    case 4
        % CNN
        experimentIdentifierString = sprintf('exp7d');
        SingleLayerCNNInit(reflist, modelfile, param);
        representations = SingleLayerCNNGetRep(modelFile, 0, noisylist);
    otherwise
        pass
end

% save the representations
save(strcat(outdir, experimentIdentifierString, '_representations'), 'representations')

%% evaluate representations
[pctList, corrList, oneList] = evaluateRepresentation(representations, ...
    noisyNameList, rateList);
%% write out the report
reportfile = strcat(outdir, repNameList{REPFLAG}, '_', experimentIdentifierString, '-', num2str(param.m), datestr(now, '_HH-MM-SS-FFF'), '.out');

%!!! write to the report file the legend entry as the first line
% (so that we can still assume the last line is the percent output)
fid = fopen(reportfile, 'a');
fprintf(fid, strcat(param.nameForLegend, '\n'));
fclose(fid);

generateEvalReport(noisyNameList, pctList, corrList, oneList, reportfile);


