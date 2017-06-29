function generateEvalReport( nameList, pctList, corrList, savePath )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
disp(['==> Generating report for ', savePath]);
fid = fopen(savePath, 'w');
for index = 1 : length(nameList)
    fprintf(fid, '--%s : %f \n', nameList{index}, pctList(index));
    for row = 1 : size(corrList, 1)
        fprintf(fid, '%s%d %s : %f \n', 'Bit #', num2str(row), ...
            'fraction of bits changed', corrList(row, index));
    end
    fprintf(fid, '%s \n', '========================================');
end

fprintf(fid, '%s \n', 'matching percentage:');
fprintf(fid, '%f ', pctList);

fclose(fid);
