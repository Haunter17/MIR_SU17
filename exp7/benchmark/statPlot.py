# ================================================
# ================================================
# usage: python3 statPlot.py <file.mat>
# ================================================
# ================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import io

mat_path = ''
try:
	mat_path = sys.argv[1]
except e:
	raise e

D = io.loadmat(mat_path)
corr_value = D.get('corrList').flatten()
ones_value = D.get('oneList').flatten()
xb_mat = D.get('xbMat')

# figure 1: single bit statistics
pos = np.arange(len(corr_value))
plt.figure()
plt.subplot(211)
plt.ylim([0, 1.0])
plt.bar(pos, corr_value, align='center', alpha=0.5)
plt.title('Correlation Statistics for Original Track: Bits Unchanged')
plt.xlabel('Bit #')
plt.ylabel('fraction of bits unchanged')
plt.autoscale(tight=True, axis='x')
plt.subplot(212)
plt.ylim([0, 1.0])
plt.bar(pos, ones_value, align='center', alpha=0.5)
plt.title('Bit Statistics for Original Track: Pct of Ones')
plt.xlabel('Bit #')
plt.ylabel('fraction of one bits')
plt.autoscale(tight=True, axis='x')
plt.tight_layout()
plt.savefig('./out/sb_stat.png', format='png')
plt.close()

# figure 2: cross bit statistics
plt.figure()
plt.imshow(xb_mat, cmap='seismic', interpolation='nearest')
plt.colorbar()
plt.title('Cross Bit Correlation')
plt.savefig('./out/xb_stat.png', format='png')
plt.close()
