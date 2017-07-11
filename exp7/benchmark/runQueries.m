function runQueries(queriesFilelist,dbfile,computeFcn, outdir,parameter)

%   Runs a set of queries on a given database, and then dumps the
%   hypothesis to file.
%
%   queriesFilelist is a text file containing the list of .wav queries
%   dbfile is a .mat file containing the fingerprint database.  This file
%      is the output of generateDB
%   outdir is the directory to dump hypothesis files
%   parameter specifies settings for CQT (must match settings in
%      precomputeQspec!)
%
%   2016-07-08 TJ Tsai ttsai@g.hmc.edu
if nargin < 5
    parameter.targetsr = 22050;
    parameter.B = 24;
    parameter.fmin = 130.81;
    parameter.fmax = 4186.01;
    parameter.gamma = 0;
    parameter.precomputeCQT = 0;
end

db = load(dbfile); % contains fingerprints, parameter, model, hopsize
fingerprints = db.fingerprints;
parameter = db.parameter;
model = db.model;
hopsize = db.hopsize;

fid = fopen(queriesFilelist);
curFileList = '';
fileIndex = 1;
curfile = fgetl(fid);
while ischar(curfile)
    curFileList{fileIndex} = curfile;
    curfile = fgetl(fid);
    fileIndex = fileIndex + 1;
end

tic;
parfor index = 1 : length(curFileList)
    curfile = curFileList{index};
    [pathstr,name,ext] = fileparts(curfile);
    disp(['Processing query ',num2str(index),': ',name]);   
    % compute hashprints    
    Q = computeQSpec(curfile,parameter);
    fpseq = computeFcn(Q,model,parameter);    
    % get match scores
    R = fastMatchFpSeq(fpseq,fingerprints);
    R(:,3) = R(:,3) * hopsize; % offsets in sec instead of hops
    % write to file
    outfile = strcat(outdir,'/',name,'.hyp');
    dlmwrite(outfile,R,'\t');
end
toc
fclose(fid);
