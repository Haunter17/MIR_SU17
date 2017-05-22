%% Task 1: run live song id system

% precompute CQT on filelist
filelist = 'audio/ref.list';
outdir = 'out/';
mkdir(outdir)
computeQSpecBatch(filelist,outdir);

% learn hashprint model
modelFile = 'out/model.mat';
param.precomputeCQT = 1;
param.precomputeCQTdir = outdir;
learnHashprintModel(filelist,modelFile,param);

% generate database of hashprints
% (note: the model file contains the list of files)
dbFile = 'out/db.mat';
generateDB(modelFile,dbFile);

% run queries
queryList = 'audio/query.list';
runQueries(queryList,dbFile,outdir);
% runQueries will generate a .hyp file in outdir for each query
% Each .hyp file will contain the list of database items 
% sorted by match score with the following format:
%    songID  matchScore offset pitchShiftInfo
% The .hyp files can then be compared to ground truth



%% Task 2: compute hashprints

% to compute hashprints on an audio file
curfile = 'audio/query.mp3';
CQTparam.B = 24;
CQTparam.fmin = 130.81;
CQTparam.fmax = 4186.01;
CQTparam.precomputeCQT = 0;
%CQTparam.precomputeCQTdir = outdir; % if precomputed
load(modelFile); % loads parameter and eigvecs

Q = computeQSpec(curfile,CQTparam); 
%Q = pitchShiftCQT(Q,-2); % pitch shift down 2 quartertones
logQspec = preprocessQspec(Q);
F = computeHashprints(logQspec,eigvecs,parameter);
%  F is a matrix of logical values containing the computed hashprint
%  bits.  The rows correspond to different bits in the fingerprint,
%  and the columns correspond to different time indices
