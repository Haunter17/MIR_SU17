function [ data ] = CNNDataHelper( fileList, parameter, labels )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    %% IO
fid = fopen(fileList);
curFileList = '';
fileIndex = 1;
curfile = fgetl(fid);

while ischar(curfile)
    curFileList{fileIndex} = curfile;
    curfile = fgetl(fid);
    fileIndex = fileIndex + 1;
end

if nargin < 3
    labels = 0 : length(curFileList) - 1;
end

%% merge data
windowSize = parameter.m;
data = {};
tic;
parfor index = 1 : length(curFileList)
    curfile = curFileList{index};
    disp(['Generating data on #',num2str(index),': ',curfile]);
    Q = computeQSpec(curfile,parameter); % struct
    logQ = preprocessQspec(Q, parameter);
    currBlock = [];
    for col = 1 : parameter.hop : size(logQ, 2) - windowSize
        sample = logQ(:, col : col + windowSize - 1);
        currBlock = [currBlock sample(:)];
    end
    currBlock = [currBlock; labels(index) * ones(1, size(currBlock, 2))];
    data{index} = currBlock;
end
    
toc
fclose(fid);
data = cell2mat(data);


end

