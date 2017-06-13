function [k] = variance_analysis(D, target)
	total_var = sum(D);
	sum_var = 0;
	k = 1;
	while k <= length(D)
		sum_var = sum_var + D(k);
		if sum_var / total_var >= target
			break
		end
		k = k + 1;
	end
end
