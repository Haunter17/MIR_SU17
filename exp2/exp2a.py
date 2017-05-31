import matplotlib.pyplot as plt
import numpy as np
import h5py


def PCA(X, target_pct = 0.99, k = -1):
	'''
		X has dimension m x n.
		Generate principal components of the data.
	'''
	# zero out the mean
	m, n = X.shape
	mu = X.mean(axis = 0).reshape(1, -1)
	X = X - np.repeat(mu, m, axis = 0)
	# unit variance
	# var = np.multiply(X, X).mean(axis = 0)
	# std = np.sqrt(var).reshape(1, -1)
	# X = np.nan_to_num(np.divide(X, np.repeat(std, m, axis = 0)))
	# svd
	U, S, V = np.linalg.svd(X.T @ X)
	if k == -1:
		# calculate target k
		total_var = sum(S ** 2)
		accum = 0.
		k = 0
		while k < len(S):
			accum += S[k] ** 2
			if accum / total_var >= target_pct:
				break
			k += 1
	# projection
	X_rot = X @ U[:, :k + 1]
	return X_rot, S ** 2, k

def PCA_analysis(D, title = 'Relative Variance Preservation', savename = 'variance.png'):
	'''
		Generate variance preservation analysis of the PCA.
	'''
	total_var = sum(D)
	D /= total_var
	plt.style.use('ggplot')
	plt.title(title)
	plt.plot(range(len(D)), D)
	plt.xlabel("Order of eigenvalue")
	plt.ylabel("Percentage of variance")
	plt.savefig(savename, format = 'png')
	plt.close()


print('==> Experiment 2a')
filepath = '../taylorswift_out/data.mat'
print('==> Loading data from {}'.format(filepath))
# benchmark
t_start = time.time()

# reading data
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))
print('--Time elapsed for loading data: {t:.2f} \
    seconds'.format(t = t_end - t_start))
del f
print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of test samples: {}'.format(X_val.shape[0]))
