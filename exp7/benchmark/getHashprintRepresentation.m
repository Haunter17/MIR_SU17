function [fingerprints] = getHashprintRepresentation(modelFile)

load(modelFile); % loads filelist, parameter, eigvecs, eigvals


fid = fopen(filelist);
curFileList = '';
fileIndex = 1;

curfile = fgetl(fid);
while ischar(curfile)
    curFileList{fileIndex} = curfile;
    curfile = fgetl(fid);
    fileIndex = fileIndex + 1;
end

fingerprints = cell(length(curFileList));

%% hashprints for original file
tic;
parfor index = 1 : length(curFileList)
    curfile = curFileList{index};
    disp(['==> Computing fingerprints on file ',num2str(index),': ',curfile]);
    Q = computeQSpec(curfile,parameter);
    logQspec = preprocessQspec(Q);
    
    % compute hashprints on original studio track
    fpseq = computeHashprints(logQspec,eigvecs,parameter);
    fingerprints{index} = fpseq;
end
toc
fclose(fid);
