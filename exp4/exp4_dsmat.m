filepath = '/pylon2/ci560sp/haunter/exp3_taylorswift_d15_1s_C1C8.mat';
disp('loading...');
load(filepath);
trainingFeatures = trainingFeatures(:, 1 : 10 : end);
trainingLabels = trainingLabels(:, 1 : 10 : end);
validationFeatures = validationFeatures(:, 1 : 10 : end);
validationLabels = validationLabels(:, 1 : 10 : end);
savepath = '/pylon2/ci560sp/haunter/exp3_small.mat';
disp('saving...');
save(savepath, 'trainingFeatures', 'trainingLabels', ...
	'validationFeatures', 'validationLabels','-v7.3');