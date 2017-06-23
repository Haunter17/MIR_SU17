import h5py
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# usage: python exp6a.py 256 0 0 0
# system arg
n_hidden = 128 # hidden layer num of features
SMALL_FLAG = 1
FAST_FLAG = 1
SYS_FLAG = 0 # 0 for bridges, 1 for supermic
try:
    n_hidden = int(sys.argv[1])
    SMALL_FLAG = int(sys.argv[2])
    FAST_FLAG = int(sys.argv[3])
    SYS_FLAG = int(sys.argv[4])
except Exception, e:
    print('-- {}'.format(e))

print('-- LSTM size: {}'.format(n_hidden))
print('-- SMALL FLAG: {}'.format(SMALL_FLAG))
print('-- FAST FLAG: {}'.format(FAST_FLAG))
print('-- SYS FLAG: {}'.format(SYS_FLAG))

print('==> Experiment 6a: LSTM memory sizes')
sys_path = '/pylon2/ci560sp/haunter/'
if SYS_FLAG:
    sys_path = '/scratch/zwang3/'
filename = 'exp3_taylorswift_d15_1s_C1C8.mat'
if SMALL_FLAG:
    filename = 'exp3_small.mat'
filepath = sys_path + filename
print('==> Loading data from {}...'.format(filepath))

# ==============================================
#               reading data
# ==============================================
# benchmark
t_start = time.time()
f = h5py.File(filepath)
X_train = np.array(f.get('trainingFeatures'))
y_train = np.array(f.get('trainingLabels'))
X_val = np.array(f.get('validationFeatures'))
y_val = np.array(f.get('validationLabels'))

t_end = time.time()
print('--Time elapsed for loading data: {t:.2f} \
        seconds'.format(t = t_end - t_start))
del f
print('-- Number of training samples: {}'.format(X_train.shape[0]))
print('-- Number of validation samples: {}'.format(X_val.shape[0]))


# ==============================================
#               RNN configs
# ==============================================
# Network Parameters
num_training_vec, total_features = X_train.shape
num_freq = 169
num_frames = int(total_features / num_freq)

max_iter = 300
print_freq = 5
if FAST_FLAG:
    max_iter = 5
    print_freq = 1

batch_size = 1000
learning_rate = 0.001
n_input = num_freq # number of sequences (rows)
n_steps = num_frames # size of each sequence (number of columns), timesteps

n_classes = int(max(y_train.max(), y_val.max()) + 1)



# ==============================================
#               RNN architecture
# ==============================================
# Transform labels into on-hot encoding form
y_train_OHEnc = tf.one_hot(y_train.copy(), n_classes)
y_val_OHEnc = tf.one_hot(y_val.copy(), n_classes)

# tf Graph input
x = tf.placeholder("float", [None, n_steps, n_input])
y = tf.placeholder("float", [None, n_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([n_classes]))
}

def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, n_steps, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# evaluation metrics
train_acc_list = []
val_acc_list = []
train_err_list = []
val_err_list = []

# saver setup
varsave_list = [weights['out'], biases['out']]
saver = tf.train.Saver(varsave_list)
save_path = './out/6amodel_{}.ckpt'.format(n_hidden)
opt_val_err = np.inf
opt_epoch = -1
step_counter = 0
max_counter = 50

# ==============================================
#               RNN training
# ==============================================
# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    y_train = sess.run(y_train_OHEnc)[:, 0, :]
    y_val = sess.run(y_val_OHEnc)[:, 0, :]

    print('==> Training the full network...')
    t_start = time.time()
    # Keep training until reach max iterations
    for epoch in range(max_iter):
        for i in range(0, num_training_vec, batch_size):
            end_ind = min(i + batch_size, num_training_vec)
            batch_x = X_train[i : end_ind]
            batch_y = y_train[i : end_ind]
            # Reshape data to get 28 seq of 28 elements
            batch_x = batch_x.reshape((-1, n_steps, n_input))
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if (epoch + 1) % print_freq == 0:
            train_acc = accuracy.eval(feed_dict={x: batch_x,\
             y: batch_y})
            train_acc_list.append(train_acc)
            val_acc = accuracy.eval(feed_dict={x: X_val.reshape((-1, n_steps, n_input)),\
             y: y_val})
            val_acc_list.append(val_acc)
            train_err = cost.eval(feed_dict={x: batch_x,\
             y: batch_y})
            train_err_list.append(train_err)
            val_err = cost.eval(feed_dict={x: X_val.reshape((-1, n_steps, n_input)),\
             y: y_val})
            val_err_list.append(val_err)      
            print("-- epoch: %d, training error %g, validation error %g"%(epoch + 1, train_err, val_err))
            # save screenshot of the model
            if val_err < opt_val_err:
                step_counter = 0    
                saver.save(sess, save_path)
                print('==> New optimal validation error found. Model saved.')
                opt_val_err, opt_epoch = val_err, epoch + 1
        if step_counter > max_counter:
            print('==> Step counter exceeds maximum value. Stop training at epoch {}.'.format(epoch + 1))
            break
        step_counter += 1      
            
    t_end = time.time()
    print('--Time elapsed for training: {t:.2f} \
        seconds'.format(t = t_end - t_start))
    # ==============================================
    # Restore model & Evaluations
    # ==============================================
    saver.restore(sess, save_path)
    print('==> Model restored to epoch {}'.format(opt_epoch))
    train_acc = accuracy.eval(feed_dict={x:X_train.reshape((-1, n_steps, n_input)), y: y_train})
    val_acc = accuracy.eval(feed_dict={x: X_val.reshape((-1, n_steps, n_input)), y: y_val})
    train_err = cost.eval(feed_dict={x: X_train.reshape((-1, n_steps, n_input)), y: y_train})
    val_err = cost.eval(feed_dict={x: X_val.reshape((-1, n_steps, n_input)), y: y_val})
    print('-- Training accuracy: {:.4f}'.format(train_acc))
    print('-- Validation accuracy: {:.4f}'.format(val_acc))
    print('-- Training error: {:.4E}'.format(train_err))
    print('-- Validation error: {:.4E}'.format(val_err))

print('-- Training accuracy --')
print([float('{:.4f}'.format(x)) for x in train_acc_list])
print('-- Validation accuracy --')
print([float('{:.4f}'.format(x)) for x in val_acc_list])
print('-- Training error --')
print([float('{:.4E}'.format(x)) for x in train_err_list])
print('-- Validation error --')
print([float('{:.4E}'.format(x)) for x in val_err_list])

print('==> Generating error plot...')
x_list = range(0, print_freq * len(train_acc_list), print_freq)
train_err_plot = plt.plot(x_list, train_err_list, 'b-', label='training')
val_err_plot = plt.plot(x_list, val_err_list, '-', color='orange', label='validation')
plt.xlabel('Number of epochs')
plt.ylabel('Cross-Entropy Error')
plt.title('Error vs Number of Epochs with {} Hidden Units'.format(n_hidden))
plt.legend(loc='best')
plt.savefig('./out/exp6a_{}.png'.format(n_hidden), format='png')
plt.close()

print('==> Finished!')
