addpath('../../cqt/');
filelist = 'something.txt';
fid = fopen(filelist);
curFileList = '';
fileIndex = 1;
curfile = fgetl(fid);

while ischar(curfile)
    curFileList{fileIndex} = curfile;
    curfile = fgetl(fid);
    fileIndex = fileIndex + 1;
end

windowSize = 20;
tic;
for index = 1 : length(curFileList)
    curfile = curFileList{index};
    disp(['Generating data on #',num2str(index),': ',curfile]);
    
    
end
toc
fclose(fid);
