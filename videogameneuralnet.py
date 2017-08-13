import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Read the csv files

training_dataset_df = pd.read_csv('sales_data_training.csv', dtype = float)
testing_dataset_df = pd.read_csv('sales_data_test.csv', dtype = float)

# Choose X and Y values for Neural Network

X_training = training_dataset_df.drop('total_earnings', axis = 1).values
Y_training = training_dataset_df[['total_earnings']].values

# Also repeat the process for testing dataset

X_testing = testing_dataset_df.drop('total_earnings', axis = 1).values
Y_testing = testing_dataset_df[['total_earnings']].values

# Select the scaler between 0 and 1

X_scaler = MinMaxScaler(feature_range = (0,1))
Y_scaler = MinMaxScaler(feature_range = (0,1))

# Scale the data

X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# Scale for testing too

X_scaled_testing = X_scaler.fit_transform(X_testing)
Y_scaled_testing = Y_scaler.fit_transform(Y_testing)

# Define Neural Network Parameters

learning_rate = 0.001
training_epoches = 100

# Define Neural Network Structure

layer_1_node = 50
layer_2_node = 100
layer_3_node = 50

# Define Neural Network Inupt and outputs

number_of_inputs = 9
number_of_outputs = 1

# Define Inupt layer

with tf.variable_scope('input'):
    X = tf.placeholder(dtype = tf.float32, shape = (None, number_of_inputs))

# Define first layer in Neural Network

with tf.variable_scope('layer_1'):
    weights = tf.get_variable(name ="weights1", shape = [number_of_inputs, layer_1_node], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases1", shape=[layer_1_node], initializer=tf.zeros_initializer())
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)

# Define second layer of Neural Network

with tf.variable_scope('layer_2'):
    weights = tf.get_variable(name="weights2", shape=[layer_1_node, layer_2_node], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases2", shape=[layer_2_node], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)

# Define third layer of Neural Network

with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name="weights3", shape=[layer_2_node, layer_3_node], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases3", shape=[layer_3_node], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)

# Define output layer of Neural Network

with tf.variable_scope('output'):
    weights = tf.get_variable(name="weights4", shape=[layer_3_node, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name="biases4", shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.matmul(layer_3_output, weights) + biases

# Define cost function that will find the accuracy of our neural network

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, number_of_outputs))
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))

# Write optimizer function that will train our neural network

with tf.variable_scope('train'):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create summary operation to log the progress of network

with tf.variable_scope('logging'):
    tf.summary.scalar("current_cost", cost)
    summary = tf.summary.merge_all()

# Save the model to the file, To save we need to first create tf.train.saver object

saver = tf.train.Saver()

# Write a session to run the about code and pass the data values

with tf.Session() as sess:
    # Ininalize global variable inintalizer to initialize all the variables

    sess.run(tf.global_variables_initializer())

    # Create log writers for training and testing seperately
    training_writer = tf.summary.FileWriter('./logs/training', sess.graph)
    testing_writer  = tf.summary.FileWriter('./logs/testing', sess.graph)

    # Run optimizer again and again to train the network for n epoches

    for epoch in range(training_epoches):
        sess.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
        #print("Session pass: {}".format(epoch))

        # Let's see the accuracy after each 5 examples run

        if epoch % 5 == 0:
            training_cost, training_summary = sess.run([cost, summary], feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            testing_cost, testing_summary = sess.run([cost, summary], feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
            print("Epoch - {} Training Cost = {} Testing Cost = {}".format(epoch, training_cost, testing_cost))

            # Write the current training status to the log files
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

    print("Training Complete Cheers!")

    final_training_cost = sess.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = sess.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

    print("Final accuracy: ")
    print("Training Cost {}".format(final_training_cost))
    print("Testing Cost {}".format(final_testing_cost))

    # Now we have created the model now its time to predict the output using testing dataset
    # Pass X testing data and run prediction operation

    Y_predicted_scaled = sess.run(prediction, feed_dict= { X: X_scaled_testing })

    # Reverse transform the predicted value
    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

    real_earnings = testing_dataset_df['total_earnings'].values[0] # Get 0 th element of array
    predicted_earnings = Y_predicted[0][0]

    print("The actual earnings of game #1 were ${}".format(real_earnings))
    print("Our Neural Network predicted ${}".format(predicted_earnings))

    # Save the model, pass session and directory you want to save the model to

    save_path = saver.save(sess, 'logs/trained_model.ckpt') # checkpoint file
    print("Model saved in {}".format(save_path))