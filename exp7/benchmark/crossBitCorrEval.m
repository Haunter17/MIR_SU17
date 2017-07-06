function [ xbcorr ] = crossBitCorrEval( rep )
%UNTITLED Summary of this function goes here
%   rep has dimension of #bit x #frames
%   returns the correlation matrix showing the correlation between ...
%   different bits
xbcorr = corrcoef(rep');
xbcorr = abs(xbcorr);

end

