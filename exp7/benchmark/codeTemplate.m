addpath('../../cqt/');
REPFLAG = 0; % 0 for baseline system

%% precompute CQT on filelist
artist = 'taylorswift';
filelist = strcat(artist, '_ref.list');
outdir = strcat(artist, '_out/');
mkdir(outdir)

nameList = {'original', '90% speed', '95% speed', '105% speed', '110% speed',...
    'generated reverberation'};
rateList = [1, 0.9, 0.95, 1.05, 1.1, 1];

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
