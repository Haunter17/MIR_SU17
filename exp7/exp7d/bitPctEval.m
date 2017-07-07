function [ one_pct ] = bitPctEval( rep )

%   returns the percentage of one's for each bit in rep
one_pct = sum(rep, 2) / size(rep, 2);

end
