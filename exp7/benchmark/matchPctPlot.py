import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')

values = []
try:
	num_comp = int(sys.argv[1])
	for i in range(num_comp):
		raw_value = sys.argv[i + 2]
		values.append([float(x) for x in raw_value.split(',')])
except e:
	raise e

colors = ['r', 'g', 'b', 'gold', 'black']
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
bar_width = 0.1
for i in range(nsuites):
	plt.subplot(4, 2, i + 1)
	plt.ylim([0, 1])
	fac = np.arange(num_comp) - num_comp / 2
	for j in range(num_comp):
		plt.bar(pos + fac[j] * bar_width, values[j], align='center', \
		width=bar_width, alpha=0.5, color=colors[j])
	plt.xticks(pos, xLabels[i])
	plt.title(groupName[i])
	plt.autoscale(tight=True, axis='x')

plt.tight_layout()
plt.savefig('./taylorswift_out/pct.png', format='png')
plt.close()
