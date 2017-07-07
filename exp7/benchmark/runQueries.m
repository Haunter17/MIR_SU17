function runQueries(queriesFilelist,dbfile,computeFcn, outdir,qparam)
% function runQueries(queriesFilelist,dbfile,outdir,qparam)
%
%   Runs a set of queries on a given database, and then dumps the
%   hypothesis to file.
%
%   queriesFilelist is a text file containing the list of .wav queries
%   dbfile is a .mat file containing the fingerprint database.  This file
%      is the output of generateDB
%   outdir is the directory to dump hypothesis files
%   qparam specifies settings for CQT (must match settings in
%      precomputeQspec!)
%
%   2016-07-08 TJ Tsai ttsai@g.hmc.edu
if nargin < 4
    qparam.targetsr = 22050;
    qparam.B = 24;
    qparam.fmin = 130.81;
    qparam.fmax = 4186.01;
    qparam.gamma = 0;
    qparam.precomputeCQT = 0;
end

load(dbfile); % contains fingerprints, parameter, model, hopsize
fid = fopen(queriesFilelist);
curfile = fgetl(fid);
count = 1;
while ischar(curfile)
    tic;
    [pathstr,name,ext] = fileparts(curfile);
    disp(['Processing query ',num2str(count),': ',name]);
    
    % compute hashprints
    Q = computeQSpec(curfile,qparam);
    logQ = preprocessQspec(Q);
    fpseq = computeFcn(logQ,model,parameter);
    
    % get match scores
    R = matchFingerprintSequence(fpseq,fingerprints);
    R(:,3) = R(:,3) * hopsize; % offsets in sec instead of hops
    
    % write to file
    outfile = strcat(outdir,'/',name,'.hyp');
    dlmwrite(outfile,R,'\t');
    curfile = fgetl(fid);
    count = count + 1;
    toc
end
fclose(fid);
