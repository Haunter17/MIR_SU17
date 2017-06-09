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
