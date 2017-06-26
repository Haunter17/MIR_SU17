import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print_freq = 5
v1 = [0.34208, 0.23126, 0.20223, 0.18834, 0.18191, 0.17678, 0.17864, 0.17478, 0.17649, 0.17908, 0.17846, 0.18053, 0.20251, 0.18986, 0.18758, 0.1906, 0.19082, 0.19142]
# v2 = []
v3 = [0.32738, 0.28121, 0.2635, 0.26579, 0.25643, 0.25796, 0.26427, 0.25847, 0.27508, 0.27392, 0.27519, 0.27638, 0.28236, 0.27163, 0.27425]
v4 = [0.37361, 0.233, 0.19628, 0.18184, 0.17396, 0.16923, 0.16909, 0.16304, 0.17084, 0.16996, 0.1653, 0.16953, 0.17458, 0.1793, 0.17697, 0.17175, 0.18682, 0.18311]
v5 = [0.28359, 0.22425, 0.20573, 0.20446, 0.20103, 0.19972, 0.20975, 0.20353, 0.20277, 0.21958, 0.20554, 0.20807, 0.23082, 0.22774, 0.2267, 0.2505]
v6 = [0.35528, 0.31769, 0.30541, 0.29783, 0.30365, 0.29241, 0.32137, 0.30878, 0.30048, 0.3326, 0.30968, 0.32452, 0.31594, 0.32356, 0.30901, 0.32051]
print('==> Generating error plot...')

x1 = range(0, print_freq * len(v1), print_freq)
# x2 = range(0, print_freq * len(v2), print_freq)
x3 = range(0, print_freq * len(v3), print_freq)
x4 = range(0, print_freq * len(v4), print_freq)
x5 = range(0, print_freq * len(v5), print_freq)
x6 = range(0, print_freq * len(v6), print_freq)

plot1 = plt.plot(x1, v1, '-', label='L=2, c=0.5')
# plot2 = plt.plot(x2, v2, '-', label='L=2, c=0.7')
plot3 = plt.plot(x3, v3, '-', label='L=2, c=0.9')
plot4 = plt.plot(x4, v4, '-', label='L=3, c=0.5')
plot5 = plt.plot(x5, v5, '-', label='L=3, c=0.7')
plot6 = plt.plot(x6, v6, '-', label='L=3, c=0.9')

plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Validation Error vs Number of Epochs')
plt.legend(loc='best')
plt.savefig('exp4i_BN.png', format='png')
plt.close()

print('==> Finished!')
