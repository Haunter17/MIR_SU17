function [ data ] = combineData( filelist, param, downSmapling )
    data = {};
    fid = fopen(filelist);
    curFileList = '';
    fileIndex = 1; 
    curfile = fgetl(fid);
    tic;
    while ischar(curfile)
        curFileList{fileIndex} = curfile;
        curfile = fgetl(fid);
        fileIndex = fileIndex + 1;
    end
    parfor count = 1 : length(curFileList)
        disp(['Loading data file ', num2str(count)]);
        curfile = curFileList{count};
        [pathstr,name,ext] = fileparts(curfile);
        cqt_file = strcat(param.precomputeCQTdir,name,'.mat');
        Qfile = load(cqt_file); % loads Q (struct)
        Q_mat = nthroot(abs(Qfile.Q.c), 3);
        Q_mat = avg_kcol(Q_mat, downSmapling)
        data{count} = Q_mat;
    end
    toc
end

function [ D ] = avg_kcol( A, k )
% average a matrix by every k columns
%   
[orig_row, orig_col] = size(A);
new_col = floor(orig_col / k) * k;
A = A(:, 1: new_col);
B = reshape(A', k, [])';
C = mean(B, 2);
D = reshape(C', [], orig_row)';
end
