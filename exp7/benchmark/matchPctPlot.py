import numpy as np
import matplotlib.pyplot as plt
import sys

plt.style.use('ggplot')

raw_value1 = []
raw_value2 = []
try:
	raw_value1 = sys.argv[1]
	raw_value2 = sys.argv[1]
except e:
	raise e

value1 = [float(x) for x in raw_value1.split(',')]
value2 = [float(x) for x in raw_value2.split(',')]

repNames = ['Hashprint', 'Variation']
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
bar_width = 0.2
for i in range(nsuites):
	plt.subplot(4, 2, i + 1)
	curr_value1 = value1[splitIndex[i]:splitIndex[i + 1]]
	curr_value2 = value2[splitIndex[i]:splitIndex[i + 1]]
	pos = np.arange(len(curr_value1))
	plt.ylim([0, 1])
	plt.bar(pos - bar_width, curr_value1, align='center', \
		width=bar_width, alpha=0.5, label=repNames[0], color=colors[0])
	plt.bar(pos, curr_value2, align='center', \
		width=0.2, alpha=0.5, label=repNames[1], color=colors[1])
	plt.xticks(pos, xLabels[i])
	plt.title(groupName[i])
	plt.autoscale(tight=True, axis='x')
	# plt.legend(loc='best')

plt.tight_layout()
plt.savefig('./taylorswift_out/pct.png', format='png')
plt.close()
