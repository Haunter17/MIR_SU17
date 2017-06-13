function samples = sampleRef(refData, param, verticalSize, horizontalSize, numSamplesPerRef)

% Returns samples, a struct containing 2 fields.
%   - samples.col is a 121 x number of samples matrix, 
%     containining training dataset for column vectors.
%   - samples.row is a 483 x number of samples matrix, 
%     containining training dataset for row vectors.
%
% Input: filelist is a list of reference files e.g. audio/ref.list
%        param - contains outdir, the location of .mat files.
%        verticalSize is the number of vertical samples
%        horizontalSize is the number of horizontal samples
%
tic;
disp(['Starting subsampling']);

columnSamples = cell(1, length(refData));
rowSamples = cell(1, length(refData));

samples = struct;

parfor refNum = 1 : length(refData)
    ref = refData{refNum};
    colBlock = zeros(verticalSize, numSamplesPerRef);
    rowBlock = zeros(horizontalSize, numSamplesPerRef);
    for i = 1:numSamplesPerRef
        randcol1 = randi(size(ref,2));
        randcol2 = randi(size(ref,2)-horizontalSize+1);
        newColSample = ref(:,randcol1);
        newRowSample = ref(randi(verticalSize), randcol2:randcol2+horizontalSize-1); 
        normCol = 0;
        if normCol
            newColSample = normc(newColSample);
        end
        colBlock(:, i) = newColSample;
        rowBlock(:, i) = newRowSample';
    end
    columnSamples{refNum} = colBlock;
    rowSamples{refNum} = rowBlock;
end

samples.col = cell2mat(columnSamples);
samples.row = cell2mat(rowSamples);
toc
