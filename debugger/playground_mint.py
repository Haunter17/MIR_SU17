# Monitor Error Distribution
def get_error_dist(features,labels,savefilename):
    """ This function evaluates the cross-entropy error for each data point in the dataset provided.
        The errors are saved into .csv file; each row has the format of [data point number, error]. 
        The function must be placed after defining cross_entropy in the code. """
        
    with open(savefilename, 'wb') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        num_vec = features.shape[0]
        for i in range(num_vec):
            x_vec = features[i].reshape(1,-1)
            y_vec = labels[i].reshape(1,-1)
            error = cross_entropy.eval(feed_dict={x: x_vec, y_: y_vec})
            spamwriter.writerow([i+1, error])
    csvfile.close()

# Example of using get_error_dist
get_error_dist(X_train,y_train,"training_error_distribution.csv")
get_error_dist(X_val,y_val,"validation_error_distribution.csv")

######################################################################################################
# Save model after training

# Change weight/bias initialization function to set variable name
def weight_variable(shape, var_name):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=var_name)

def bias_variable(shape, var_name):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=var_name)

# Example of initializing weight
W_conv1 = weight_variable([121, 1, 1, 12], "W_conv1")
b_conv1 = bias_variable([12], "b_conv1")

W_fc = weight_variable([16*12, 71], "W_fc")
b_fc = bias_variable([71], "b_fc")

# Add variables that we want to save to a collection
# The name of the collection here is 'vars'
tf.add_to_collection('vars', W_conv1)
tf.add_to_collection('vars', b_conv1)
tf.add_to_collection('vars', W_fc)
tf.add_to_collection('vars', b_fc)

#Create a saver object which will save all the variables
saver = tf.train.Saver()

# Save model after training
sess.run(y_conv,feed_dict={x: X_train, y_: y_train})
saver.save(sess, 'my-model')

######################################################################################################
# Example of code to restore model from file

sess = tf.Session()
new_saver = tf.train.import_meta_graph('my-model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
for v in all_vars:
    # Print information of the tensor object (name, shape, etc.)
    print(v)
    # Print value stored in the tensor object (weight matrix)
    v_ = sess.run(v)
    print(v_)
