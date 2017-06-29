import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot')

value = [1.000000, 0.498568, 0.506024, 0.578511, 0.521901, 0.990974, 0.994887, 0.997284, 0.882582, 0.777484, 0.738703, 0.585472, 0.622477, 0.622255, 0.585837, 0.730731, 0.793328, 0.790557, 0.775457, 0.495966, 0.574137, 0.642166, 0.704796, 0.750554, 0.790817, 0.831332, 0.870550, 0.996986, 0.512913, 0.534327, 0.571202, 0.625168, 0.689062, 0.755214, 0.813235, 0.995361, 0.549406, 0.598343, 0.662568, 0.724493, 0.787837, 0.840404, 0.883413, 0.997422]
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
	plt.bar(pos, curr_value, align='center', alpha=0.5)
	plt.xticks(pos, xLabels[i])
	plt.title(groupName[i])

plt.tight_layout()
plt.savefig('./taylorswift_out/pct.png', format='png')
plt.close()
