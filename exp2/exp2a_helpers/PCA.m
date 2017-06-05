function [U, S] = PCA(artist, direction, data, threshold)
% Taking sampled data and perform principal component analysis
mkdir(strcat(artist, '_out/PCAFig/'))
%% normalize data
data = data - repmat(mean(data, 2), 1, size(data, 2));
%% SVD for PCA
sigma = data * data' / size(data, 2);
disp(['==> Starting PCA for ', direction, ' vectors...']);
[U, S, ~] = svd(sigma);
%% Display eigenvectors
disp(['==> Plotting principal components']);
% colormap jet;
% uPlot = figure;
% imagesc(U);
% saveas(uPlot, strcat(artist, '_out/PCAFig/',direction,'PCAFull.png'));

evPlot = figure;
for i = 1 : 6
    subplot(2, 3, i);
    stem(1:size(U, 1), U(:, i));
    title(strcat(direction, ' eigenvector #', num2str(i)));
end
saveas(evPlot, strcat(artist, '_out/PCAFig/', direction, 'Eigenvectors.png'));
%% Display eigenvalues
E = S .^ 2;
k = variance_analysis(diag(E), threshold);
disp(['-- Number of desired components for ', direction, ' vectors is ', int2str(k)]);

Var = E / sum(sum(E));
disp(['-- First component takes up ', num2str(Var(1, 1)), ' of variance']);
disp(['==> Plotting percentage of variance of eigenvalues']);
% hSingularStem = figure;
% stem(1:size(Var, 1), diag(Var));
% saveas(hSingularStem, strcat(artist, '_out/PCAFig/', direction, 'VarEigenvalAll.png'));
hSingularStemSub = figure;
stem(diag(Var(1:10, 1:10)));
axis([1 10 0 1.1 * Var(2, 2)]);
title(strcat('Relative Variance of Top 10 ', direction, ' Eigenvalues'));
saveas(hSingularStemSub, strcat(artist, '_out/PCAFig/', direction, 'VarEigenvalSub.png'));
