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
prompt = 'Do you want to run the minibenchmark?\n (0 = no, 1 = yes)\n';
MINIFLAG = input(prompt);
prompt = 'Do you want to run the test benchmark?\n (0 = no, 1 = yes)\n';
TESTFLAG = input(prompt);
prompt = 'Do you want to run the validation benchmark\n (0 = no, 1 = yes)\n';
VALFLAG = input(prompt);
prompt = 'What is the system flag index?\n (1 = hashprint, 2 = randomized, 3 = AE)\n';
REPFLAG = input(prompt);
prompt = 'What is the name of the model file? (Do not include ".mat" in the input)\n';
modelName = input(prompt, 's');

if MINIFLAG
    prompt = 'What is the name of the output report file? (Do not include ".out" in the input)\n';
    reportName = input(prompt, 's');
end

repNameList = {'hashprint', 'randomized', 'AE'};

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
    queryList = strcat(artist, '_query.list');
    runQueries(queryList, dbFile, computeFcn, outdir);
    
    %% run MRR
    q2rList = strcat(artist, '_querytoref.list');
    disp(['Calculating MRR for ', artist, ' test queries']);
    MRR = calculateMRR(q2rList, strcat(artist, '_query'), outdir);
    disp(['MRR is ', num2str(MRR)]);
end

if VALFLAG
    %% generate database
    dbFile = strcat(outdir, modelName, '_db.mat');
    if exist(dbFile, 'file') == 0
        generateDB(modelFile, computeFcn, reflist, dbFile);
    end
    disp(['Database saved at ', dbFile]);
    
    %% run validation queries
    queryList = strcat(artist, '_val.list');
    runQueries(queryList, dbFile, computeFcn, outdir);
    
    %% run MRR
    q2rList = strcat(artist, '_valtoref.list');
    disp(['Calculating MRR for ', artist, ' validation queries']);
    MRR = calculateMRR(q2rList, strcat(artist, '_val'), outdir);
    disp(['MRR is ', num2str(MRR)]);
end
