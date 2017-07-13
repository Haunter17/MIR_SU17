addpath('../../cqt/');
addpath('../exp7d/');
addpath('../autoencoder/');
%% Parallel computing setup
curPool = gcp('nocreate'); 
if (isempty(curPool))
    myCluster = parcluster('local');
    numWorkers = myCluster.NumWorkers;
    % create a parallel pool with the number of workers in the cluster`
    pool = parpool(ceil(numWorkers * 0.75));
end

%% input setup
repNameList = {'hashprint', 'randomized', 'AE'};

%% precompute CQT on reflist
reflist = strcat('./audio/', artist, '_ref.list');
noisylist = strcat('./audio/', artist, '_noisy.list');
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

param.precomputeCQT = 0;
param.precomputeCQTdir = outdir;
computeQSpecBatch(reflist,outdir, param);
if MINIFLAG
    computeQSpecBatch(noisylist, outdir, param);
end

%% learn models and generate representations
param.m = -1;
modelFile = strcat(outdir, modelName, '.mat');
computeFcn = 0;
% switch for different representations
switch REPFLAG
    case 1
        param.m = 20;
	prompt = 'Enter the number of context frames (default is 20): \n';
	param.m = input(prompt);
        learnHashprintModel(reflist, modelFile, param);
        computeFcn = @computeHashprints;
    case 2
        param.numFiltersList = [256];
        RandomProjectionModelInit(reflist, modelFile, param);
        computeFcn = @computeAlphaHashprints;
    case 3
        initAEModel(modelFile, param);
        computeFcn = @computeAErep;
    otherwise
        pass
end

if MINIFLAG
    %% get representations
    representations = getRepresentations(modelFile, computeFcn, noisylist);

    %% evaluate representations
    [pctList, corrList, oneList, xbMat] = evaluateRepresentation(representations, ...
        rateList);
    report_prefix = strcat(outdir, reportName);
    outfile = strcat(report_prefix, datestr(now, '_HH-MM-SS-FFF'), '.out');
    matfile = strcat(report_prefix, '_out.mat');
    generateEvalReport(noisyNameList, pctList, corrList, oneList, xbMat, ...
        outfile, matfile);
end

if TESTFLAG
    %% generate database
    dbFile = strcat(outdir, modelName, '_db.mat');
    if exist(dbFile, 'file') == 0
        generateDB(modelFile, computeFcn, reflist, dbFile);
    end
    disp(['Database saved at ', dbFile]);
    
    %% run test queries
    queryList = strcat('./audio/', artist, '_query.list');
    runQueries(queryList, dbFile, computeFcn, outdir);
    
    %% run MRR
    q2rList = strcat('./audio/', artist, '_querytoref.list');
    disp(['Calculating MRR for ', artist, ' test queries']);
    testMRR = calculateMRR(q2rList, strcat(artist, '_query'), outdir);
    disp(['Test MRR is ', num2str(testMRR)]);
end

if VALFLAG
    %% generate database
    dbFile = strcat(outdir, modelName, '_db.mat');
    if exist(dbFile, 'file') == 0
        generateDB(modelFile, computeFcn, reflist, dbFile);
    end
    disp(['Database saved at ', dbFile]);
    
    %% run validation queries
    queryList = strcat('./audio/', artist, '_val.list');
    runQueries(queryList, dbFile, computeFcn, outdir);
    
    %% run MRR
    q2rList = strcat('./audio/', artist, '_valtoref.list');
    disp(['Calculating MRR for ', artist, ' validation queries']);
    valMRR = calculateMRR(q2rList, strcat(artist, '_val'), outdir);
    disp(['Validation MRR is ', num2str(valMRR)]);
end
