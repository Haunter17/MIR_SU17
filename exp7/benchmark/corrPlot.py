# ================================================
# ================================================
# usage: python3 corrPlot.py '1.0, 0.9, 0.8, 0.7 ...'
# ================================================
# ================================================

import numpy as np
import matplotlib.pyplot as plt
import sys
plt.style.use('ggplot')

raw_value = []
try:
	raw_value = sys.argv[1]
except e:
	raise e

value = [float(x) for x in raw_value.split(',')]

pos = np.arange(len(value))
plt.ylim([0, 1.0])
plt.bar(pos, value, align='center', alpha=0.5)
plt.title('Correlation Statistics for Original Track')
plt.xlabel('Bit #')
plt.ylabel('fraction of bits changed')

plt.autoscale(tight=True, axis='x')
plt.savefig('./taylorswift_out/corr.png', format='png')
plt.close()
