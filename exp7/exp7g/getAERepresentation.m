function [fingerprints] = getAERepresentation(modelFile, flist)
% note that flist is different from filelist, which is loaded by modelFile
load(modelFile); % loads W, b
if nargin < 2
    flist = filelist;
end

fingerprints = {};
fid = fopen(flist);
count = 1;
curfile = fgetl(fid);

%% hashprints for original file
while ischar(curfile)
    tic;
    disp(['==> Computing fingerprints on file ',num2str(count),': ',curfile]);
    Q = computeQSpec(curfile,parameter);
    logQspec = preprocessQspec(Q);
    
    % compute hashprints on original studio track
    fpseq = computeAErep(logQspec, W, b, parameter);
    fingerprints{count} = fpseq;
    
    % compare bit match
    count = count + 1;
    curfile = fgetl(fid);
    toc
end
fclose(fid);
