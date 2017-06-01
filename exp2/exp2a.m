%% use the data with 121 features
load('../taylorswift_out/data.mat');
X_train = trainingFeatures';
y_train = trainingLabels';
disp(['-- Number of training samples ', int2str(size(X_train, 1))]);

% %% perform PCA on horizontal and vertical features
% disp(['==> PCA on frequency...']);
% [~, S_freq] = PCA(X_train);
% k_freq = variance_analysis(diag(S_freq) .^ 2, 0.99); % 55
% disp(['Number of desired components for frequency (column) vectors is ', int2str(k_freq)]);

mult_factor = 483;
disp(['PCA on temporal...']);
X_train_sub = X_train(1: mult_factor, :);
[~, S_temp] = PCA(X_train_sub');

k_temp = variance_analysis(diag(S_temp) .^ 2, 0.99); % 44
disp(['Number of desired components for row (temporal) vectors is ', int2str(k_temp)]);

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

function [U, S] = PCA(data)
	%% normalize data
	data = data - repmat(mean(data, 1), size(data, 1), 1);
	%% SVD for PCA
	sigma = data' * data / size(data, 1);
	disp(['Starting PCA']);
	[U, S, ~] = svd(sigma);
end
