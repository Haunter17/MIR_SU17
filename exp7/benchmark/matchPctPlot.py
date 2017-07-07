
'''
	usage:
	python3 matchPctPlot.py <#systems> <name list> <file 1> <file 2> ...
'''

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import io

plt.style.use('ggplot')

values = []
try:
	num_comp = int(sys.argv[1])
	nameList = sys.argv[2].split(',')
	for i in range(num_comp):
		mat_path = sys.argv[i + 3]
		D = io.loadmat(mat_path)
		temp = D.get('pctList')
		values.append(temp.flatten())
except e:
	raise e

colors = ['r', 'orange', 'yellow', 'g', 'b']
splitIndex = [1, 5, 11, 15, 20, 28, 36, 44]
groupName = ['Stretch/Compress', 'Amplitude', 'Pitch Shift', 'Reverberation', \
'Crowd Noise', 'Restaurant Noise', 'AWGN']

xLabels = [
	['5% fast', '10% fast', '5% slow', '10% slow'],
	['-15%', '-10%', '-5%', '+5%', '+10%', '+15%'],
	['-1', '-0.5', '+0.5', '+1'],
	['dkw', 'gal', 'shan', 'shan', 'generated'],
	['-15dB', '-10dB', '-5dB', '0dB', '5dB', '10dB', '15dB', '100dB'],
	['-15dB', '-10dB', '-5dB', '0dB', '5dB', '10dB', '15dB', '100dB'],
	['-15dB', '-10dB', '-5dB', '0dB', '5dB', '10dB', '15dB', '100dB']
]

nsuites = 7
bar_width = 0.5 / num_comp
fig = plt.figure()

cache = [None for i in range(num_comp)]

for i in range(nsuites):
	plt.subplot(4, 2, i + 1)
	plt.ylim([0, 1])
	fac = np.arange(num_comp) - num_comp / 2
	for j in range(num_comp):
		curr_val = values[j][splitIndex[i]:splitIndex[i + 1]]
		pos = np.arange(len(curr_val))
		cache[j] = plt.bar(pos + fac[j] * bar_width, curr_val, align='center', \
		width=bar_width, alpha=0.7, color=colors[j], label='hhh')
	plt.xticks(pos, xLabels[i])
	plt.title(groupName[i])
	plt.autoscale(tight=True, axis='x')

plt.tight_layout()
fig.legend(cache, nameList, 'lower right')
plt.savefig('./out/pct.png', format='png')
plt.close()
