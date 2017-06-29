function [ corr_pct ] = correlationEval( rep )
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here
rep_pad_back = [rep zeros(size(rep, 1), 1)];
rep_pad_front = [zeros(size(rep, 1), 1) rep];
diff = xor(rep_pad_back, rep_pad_front);
clear rep_pad_back rep_pad_front
diff = double(diff(:, 2 : size(diff, 2) - 1));
corr_pct = sum(diff, 2) / size(diff, 2);

% for index = 1 : size(corr_pct, 1)
%     disp(['-- Bit #', num2str(index), ' correlation statistics: ',...
%         num2str(corr_pct(index))]);
% end

end

