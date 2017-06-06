function PCABatch(artist, refData, parameter, configFile)
	
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

if isfield(parameter,'deltaDelay')==0
    parameter.deltaDelay=16;
end
if isfield(parameter,'precomputeCQT')==0
    parameter.precomputeCQT = 0;
end

%% random sampling
load(configFile); %loading verticalSize, horizontalSize, numSampPerRef, N, M, numFeatures
samples = sampleRef(refData, parameter, verticalSize, horizontalSize, numSampPerRef);


%% perform PCA on horizontal and vertical features
% Save visualization of eigenvectors and singular values
disp(['visualizing PCA on temporal samples']);
[pcaRow, SRow] = PCA(artist, 'row', normc(samples.row), 0.99);

disp(['visualizing PCA on frequency samples']);
[pcaCol, SCol] = PCA(artist, 'col', normc(samples.col), 0.99);
