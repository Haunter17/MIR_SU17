import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print_freq = 5
v1 = [3.8663, 2.7087, 2.1675, 1.7794, 1.6315, 1.5546, 1.4928, 1.4335, 1.4304, 1.3565, 1.3241, 1.3334, 1.2859, 1.2038, 1.1754, 1.1661, 1.1911, 1.1734, 1.149, 1.1427, 1.1416, 1.1706, 1.1874, 1.2955, 1.2107, 1.2103, 1.2398, 1.2419, 1.4037, 1.35, 1.3952]
v2 = [4.193, 2.5129, 1.7014, 1.5507, 1.3858, 1.3875, 1.316, 1.317, 1.2562, 1.2573, 1.3694, 1.2628, 1.4537, 1.2633, 1.3433, 1.2246, 1.326, 1.3859, 1.3608, 1.3979, 1.4803, 1.3856, 1.3869, 1.4648, 1.4334, 1.482]
v3 = [4.0027, 3.1892, 2.0527, 1.4583, 1.4189, 1.6027, 1.5252, 1.6599, 1.6948, 1.8057, 1.568, 1.8141, 1.8448, 2.5264, 1.9842]
v4 = [4.0107, 4.0522, 3.3079, 2.3271, 1.4964, 1.4242, 1.6782, 2.0185, 1.7286, 1.8088, 2.368, 2.1423, 2.2795, 2.3781, 2.4437, 2.4981]
v5 = [4.1555, 13.307, 1.8041, 1.343, 1.4752, 1.4644, 1.8592, 2.022, 4.3608, 1.9278, 1.9348, 2.019, 2.2633, 2.3876]
print('==> Generating error plot...')

x1 = range(0, print_freq * len(v1), print_freq)
x2 = range(0, print_freq * len(v2), print_freq)
x3 = range(0, print_freq * len(v3), print_freq)
x4 = range(0, print_freq * len(v4), print_freq)
x5 = range(0, print_freq * len(v5), print_freq)
# x6 = range(0, print_freq * len(v6), print_freq)

plot1 = plt.plot(x1, v1, '-', label='#hidden = 128')
plot2 = plt.plot(x2, v2, '-', label='#hidden = 256')
plot3 = plt.plot(x3, v3, '-', label='#hidden = 512')
plot4 = plt.plot(x4, v4, '-', label='#hidden = 768')
plot5 = plt.plot(x5, v5, '-', label='#hidden = 1024')

plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Validation Error vs Number of Epochs')
plt.legend(loc='best')
plt.savefig('exp6a.png', format='png')
plt.close()

print('==> Finished!')
