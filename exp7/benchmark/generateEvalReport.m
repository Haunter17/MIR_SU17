function generateEvalReport( nameList, value, savePath )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
disp(['==> Generating report for ', savePath]);
fid = fopen(savePath, 'w');
for index = 1 : length(nameList)
    fprintf(fid, '--%s : %f', nameList{index}, value(index));
end
fclose(fid);
