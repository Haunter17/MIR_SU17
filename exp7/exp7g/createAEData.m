addpath('../../cqt/');
addpath('../benchmark/');

%% Parallel computing setup
curPool = gcp('nocreate'); 
if (isempty(curPool))
    myCluster = parcluster('local');
    numWorkers = myCluster.NumWorkers;
    % create a parallel pool with the number of workers in the cluster`
    pool = parpool(ceil(numWorkers * 0.75));
end

%% config
artist = 'taylorswift';
reflist = strcat('../benchmark/', artist, '_ref.list');
outdir = strcat('../benchmark/', artist, '_out/');
parameter.precomputeCQT = 1;
parameter.precomputeCQTdir = outdir;
savename = strcat(outdir, 'AEdata.mat');

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

%% IO
fid = fopen(reflist);
curFileList = '';
fileIndex = 1;
curfile = fgetl(fid);

while ischar(curfile)
    curFileList{fileIndex} = curfile;
    curfile = fgetl(fid);
    fileIndex = fileIndex + 1;
end

%% merge data
windowSize = parameter.m;
data = {};
tic;
parfor index = 1 : length(curFileList)
    curfile = curFileList{index};
    disp(['Generating data on #',num2str(index),': ',curfile]);
    Q = computeQSpec(curfile,parameter); % struct
    logQ = preprocessQspec(Q);
    currBlock = [];
    for col = 1 : parameter.hop : size(logQ, 2) - windowSize
        sample = logQ(:, col : col + windowSize - 1);
        currBlock = [currBlock sample(:)];
    end
    data{index} = currBlock;
end
    
toc
fclose(fid);

save(savename, 'data', '-v7.3');
