function [ pct ] = compareHashprints( fpseq1, fpseq2, rate )
%
%   fpseq1 is the representation for the original soundtrack
%   fpseq2 is the representation of a noisy soundtrack
%   rate is the speed ratio between fpseq1 and fpseq2. value higher than 1
%   means fpseq2 is slower than fpseq1
% ========================================================================
%   pct is the percentage of bits that match between fpseq1 and fpseq2
%   

if nargin < 3
    rate = 1;
end

numMismatch = 0;
% same tempo
if rate == 1
    XORBlock = xor(fpseq1, fpseq2);
    numMismatch = sum(sum(XORBlock, 1));
else
    for col = 1 : size(fpseq1, 2)
        int1 = fpseq1(:, col);
        % round to the nearest column number
        int2 = fpseq2(:, min(round(col * rate, 0), size(fpseq2, 2)));
        currMiss = sum(xor(int1, int2));
        numMismatch = numMismatch + currMiss;
    end
    
end
pct = 1 - numMismatch / (size(fpseq1, 1) * size(fpseq1, 2));        
end

