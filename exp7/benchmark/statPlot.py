# ================================================
# ================================================
# usage: python3 statPlot.py '0.5, 0.3, 0.8...' '0.6, 0.2, 0.3...'
# ================================================
# ================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use('ggplot')

raw_corr_value = []
raw_ones_value = []
try:
	raw_corr_value = sys.argv[1]
	raw_ones_value = sys.argv[2]
except e:
	raise e

corr_value = [float(x) for x in raw_corr_value.split(',')]
ones_value = [float(x) for x in raw_ones_value.split(',')]

pos = np.arange(len(corr_value))
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

plt.savefig('./taylorswift_out/stat.png', format='png')
plt.close()
