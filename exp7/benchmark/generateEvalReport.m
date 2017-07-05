function generateEvalReport( nameList, pctList, corrList, oneList, savePath )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
disp(['==> Generating report for ', savePath]);
fid = fopen(savePath, 'w');

%% correlation report
for row = 1 : size(corrList, 1)
        fprintf(fid, '-> Bit %d : fraction of bits remained %f \n', ...
            row, corrList(row));
end
fprintf(fid, '%s \n', '========================================');

%% bit percentage report
for row = 1 : size(oneList, 1)
        fprintf(fid, '-> Bit %d : fraction of ones %f \n', ...
            row, oneList(row));
end
fprintf(fid, '%s \n', '========================================');

%% comparison report
for index = 2 : length(nameList)
    fprintf(fid, '--%s : %f \n', nameList{index}, pctList(index));    
end

%% data printout
fprintf(fid, '%s \n', 'correlation percentage:');
fprintf(fid, '%f, ', corrList);
fprintf(fid, '%s \n', '');
fprintf(fid, '%s \n', 'ones percentage:');
fprintf(fid, '%f, ', oneList);
fprintf(fid, '%s \n', '');
fprintf(fid, '%s \n', 'matching percentage:');
fprintf(fid, '%f, ', pctList);
fprintf(fid, '%s \n', '');
fclose(fid);
