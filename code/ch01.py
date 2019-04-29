

# load the MNIST data
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# define the network
import network
# the list defines the number of nodes in layers 0, 1, 2.
net = network.Network([784, 30, 10])

#stochastic gradient descent to learn from the MNIST data
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.0, test_data=test_data)
