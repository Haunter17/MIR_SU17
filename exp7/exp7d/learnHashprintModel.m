function learnHashprintModel(filelist,saveFilename,parameter)
% learnHashprintModel(filelist,saveFilename,parameter)
% 
%   Learns the hashprint model based on a set of files, and writes
%   the model to the specified .mat file.
%
%   filelist is a text file specifying the audio files to process.
%   saveFilename is the name of the .mat file to save the model to.
%   parameter specifies arguments for the learning
%      parameter.m specifies the number of context frames used in 
%         the time-delay embedding
%      parameter.tao specifies the gap (in frames) between consecutive 
%         frames in the time-delay embedding.
%      parameter.hop specifies the hop size (in frames) between consecutive
%         time delay embedded features
%      parameter.numFeatures specifies the number of bits in the fingerprint.
%      parameter.deltaDelay is the delay (in hops) used in the 
%         delta feature computation.
%      parameter.precomputeCQT is 1 or 0, specifying if the CQT has been
%         precomputed.  
%      parameter.precomputeCQTdir specifies the directory containing the 
%         precomputed CQT .mat files.
%
% 2016-07-08 TJ Tsai ttsai@g.hmc.edu
if nargin<3
    parameter=[];
end
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

%% Compute accumulated covariance matrix
fid = fopen(filelist);
curfile = fgetl(fid);
count = 0;
covAccum = 0; % dimension is unknown, so initialize as a single value
while ischar(curfile)
    tic;
    disp(['Computing covariance matrix on #',num2str(count+1),': ',curfile]);
    [cov,nsamples] = getTDECov(curfile,parameter);
    if nsamples > 1
        covAccum = covAccum + cov;
        count = count + 1;
    end
    curfile = fgetl(fid);
    toc
end
fclose(fid);

%% Compute eigenvectors
tic;
disp(['Computing eigenvectors on accumulated covariance matrix ... ']);
avgCovMatrix = covAccum/count;
[eigvecs,eigvals] = eig(avgCovMatrix);
eigvals = diag(eigvals);
eigvecs = fliplr(eigvecs); % order from highest eigenvalue to lowest
eigvecs = eigvecs(:,1:parameter.numFeatures);
eigvals = flipud(eigvals);
toc

%% Save to file
disp(['Saving hashprint models to file']);
save(saveFilename,'filelist','parameter','eigvecs','eigvals');

