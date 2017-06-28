addpath('../../cqt/');
REPFLAG = 0; % 0 for baseline system

%% precompute CQT on filelist
artist = 'taylorswift';
filelist = strcat(artist, '_ref.list');
outdir = strcat(artist, '_out/');
mkdir(outdir)

nameList = {'original', '90% speed', '95% speed', '105% speed', '110% speed',...
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

rateList = ones(length(nameList));
rateList(2) = 0.9;
rateList(3) = 0.95;
rateList(4) = 1.05;
rateList(5) = 1.1;

param.precomputeCQT = 1;
param.precomputeCQTdir = outdir;
computeQSpecBatch(filelist,outdir);

%% learn models and generate representations
param.precomputeCQT = 1;
param.precomputeCQTdir = outdir;
modelFile = strcat(outdir, 'model.mat');

% switch for different representations
switch REPFLAG
    case 0
        learnHashprintModel(filelist,modelFile,param);
        representations = getHashprintRepresentation(modelFile);
    otherwise
        pass
end

%% evaluate representations
evaluateRepresentation(representations, nameList, rateList);
