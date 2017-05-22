function createDatasetBatch(filelist, filename, outdir)

% Call pitchProfilePartial to compute energy of each pitch class for 
% each track in filelist. This is for test files, because training is done
% only on songs without pitch shifts.

fid = fopen(filelist);
curfile = fgetl(fid);
trainingSet = [];
testingSet = [];

while ischar(curfile)
    [pathstr,name,ext] = fileparts(curfile);
    cqt_file = strcat(outdir,name,'.mat');
    Qfile = load(cqt_file); % loads Q (struct)
    
    trainVec, testVec = createDataset(Qfile, outdir, name);
    trainingSet = cat(1, trainingSet, trainVec);
    testingSet = cat(1, testingSet, testVec);
    
    curfile = fgetl(fid);
end

save(filename, 'trainingSet', 'testingSet');
