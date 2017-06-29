function generateEvalReport( nameList, pctList, corrList, savePath )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
disp(['==> Generating report for ', savePath]);
fid = fopen(savePath, 'w');
for index = 2 : length(nameList)
    fprintf(fid, '--%s : %f \n', nameList{index}, pctList(index));
    for row = 1 : size(corrList, 1)
        fprintf(fid, '-> %d : %f \n', num2str(row), corrList(row, index));
    end
    fprintf(fid, '%s \n', '========================================');
end

fprintf(fid, '%s \n', 'matching percentage:');
fprintf(fid, '%f ', pctList);
fprintf(fid, '%s \n', '');
fclose(fid);
