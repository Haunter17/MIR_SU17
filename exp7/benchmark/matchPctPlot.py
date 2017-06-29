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
for i in range(nsuites):
	plt.subplot(4, 2, i + 1)
	curr_value = value[splitIndex[i]:splitIndex[i + 1]]
	pos = np.arange(len(curr_value))
	plt.ylim([0, 1])
	plt.bar(pos, curr_value, align='center', width=0.2, alpha=0.5)
	plt.xticks(pos, xLabels[i])
	plt.title(groupName[i])
	plt.autoscale(tight=True, axis='x')

plt.tight_layout()
plt.savefig('./taylorswift_out/pct.png', format='png')
plt.close()
