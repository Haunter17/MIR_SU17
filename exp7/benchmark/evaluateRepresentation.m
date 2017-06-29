function [pctList] = evaluateRepresentation( representations, nameList, rateList)
%
% representations - a cell of logical matrix (representations) of each audio file
% nameList - a list of strings representing the name of each audio file
% rateList - a list of floating point numbers representing the ratio of
% speed to the original soundtrack (a value greater than 1 means slower)
%
% =========================================================================
% print the matching percentage of each noisy version compared to the
% original clean soundtrack
%

orig = representations{1};
pctList = ones(1, length(representations));
for i = 2 : length(representations)
    rep = representations{i};
    pct = compareHashprints(orig, rep, rateList(i));
    pctList(i) = pct;
end
