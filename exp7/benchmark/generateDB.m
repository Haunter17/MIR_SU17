function generateDB(modelFile, computeFcn, flist, saveFilename)
% generateDB(modelFile,saveFilename)
%
%   generateDB computes hashprints on a set of studio tracks and saves 
%   the fingerprint database to the specified file.
%
%    modelFile: file specifying the hashprint model.  This is the file
%      produced by learnHashprintModel.  Note that this model file also
%      specifies the list of studio tracks to process.
%    saveFilename: the name of the file to save the fingerprint database
%
%   Note that, in addition to computing hashprints on the original studio
%   recordings, generateDB also computes hashprints on pitch-shifted
%   versions.
%
% 2016-07-08 TJ Tsai ttsai@g.hmc.edu

maxPitchShift = 4;
model = load(modelFile);
parameter = model.parameter;

fingerprints = {};
idx2file = {};

fid = fopen(filelist);
curFileList = '';
fileIndex = 1;
curfile = fgetl(fid);
while ischar(curfile)
    curFileList{fileIndex} = curfile;
    curfile = fgetl(fid);
    fileIndex = fileIndex + 1;
end

parfor index = 1 : length(curFileList)
    curfile = curFileList{index};
    disp(['Computing fingerprints on file ',num2str(index),': ',curfile]);
    Q = computeQSpec(curfile,parameter);
    logQspec = preprocessQspec(Q);
    
    % compute hashprints on original studio track
    origfpseq = computeFcn(logQspec,model,parameter);
    fpseqs = false(parameter.numFeatures,size(origfpseq,2),2*maxPitchShift+1);
    fpseqs(:,:,1) = origfpseq;
    
    % compute hashprints on pitch-shifted versions
    for i=1:maxPitchShift % shift up
	    logQspec = preprocessQspec(pitchShiftCQT(Q,i));
        fpseqs(:,:,i+1) = computeFcn(logQspec,model,parameter);
    end
    for i=1:maxPitchShift % shift down
	    logQspec = preprocessQspec(pitchShiftCQT(Q,-1*i));
        fpseqs(:,:,i+1+maxPitchShift) = computeFcn(...
            logQspec,model,parameter);
    end
    fingerprints{index} = fpseqs;
    idx2file{index} = curfile;
    
    % compute hop size -- hack!
    hopsize = Q.xlen/(22050*size(Q.c,2))*3*parameter.hop;
end
fclose(fid);

disp(['Saving fingerprint database to file']);
save(saveFilename,'flist','parameter','model',...
    'fingerprints','idx2file','hopsize');
