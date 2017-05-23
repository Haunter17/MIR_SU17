function createDatasetBatch(filelist, filename, outdir)

% Call pitchProfilePartial to compute energy of each pitch class for 
% each track in filelist. This is for test files, because training is done
% only on songs without pitch shifts.

fid = fopen(filelist);
curfile = fgetl(fid);
trainingSet = [];
testSet = [];
label = 0;

while ischar(curfile)
	disp(['-- Generating data matrix on file ', int2str(label)]);
    [pathstr,name,ext] = fileparts(curfile);
    cqt_file = strcat(outdir,name,'.mat');
    Qfile = load(cqt_file); % loads Q (struct)
    QMat = abs(Qfile.Q.c);
    
    [trainVec, testVec] = createDataset(QMat, label);
    trainingSet = cat(1, trainingSet, trainVec);
    testSet = cat(1, testSet, testVec);
    
    curfile = fgetl(fid);
    label = label + 1;
end

disp(['==> Shuffling data...'])
trainingSet = trainingSet(randperm(size(trainingSet, 1)), :)';
testSet = testSet(randperm(size(testSet, 1)), :)';
save(filename, 'trainingSet', 'testSet', '-v7.3');
