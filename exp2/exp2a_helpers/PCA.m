function [U, S] = PCA(data)
	%% normalize data
	data = data - repmat(mean(data, 1), size(data, 1), 1);
	%% SVD for PCA
	sigma = data' * data / size(data, 1);
	disp(['Starting PCA']);
	[U, S, ~] = svd(sigma);
end
