
Structure is input -> first layer -> second layer -> output
                728       n_per            n_per        10
    Non linearity of relu between the layers.

Model code:

W_fc1 = weight_variable([784, n_per_layer])
b_fc1 = bias_variable([n_per_layer])
h_fc1 = tf.nn.relu(tf.matmul(x, W_fc1) + b_fc1)


W_fc2 = weight_variable([n_per_layer, n_per_layer])
b_fc2 = bias_variable([n_per_layer])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

W_fc3 = weight_variable([n_per_layer, 10])
b_fc3 = bias_variable([10])
y = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)


Results:
    fc_best_100      100 neurons in the middle layer. Gets 97.4% accuracy
    fc_best_1000     1000 neurons in the middle layer. Gets 98.15% accuracy.