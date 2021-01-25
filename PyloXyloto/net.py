from PyloXyloto.visualization import *
from PyloXyloto.utils import *


class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_derivative = None
        self.error_per_epoch = []
        self.accuracy_per_epoch = []

    # add layer to network
    def add_layer(self, layer):
        self.layers.append(layer)

    # set loss
    def set_loss(self, loss, loss_derivative):
        self.loss = loss
        self.loss_derivative = loss_derivative

    # train the network
    def train(self, x_train, y_train, epochs, learning_rate):
        # sample dimension first
        samples = len(x_train)

        # training loop
        for i in range(epochs):
            err = 0
            out_list, predicted = [], []
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                    out_list = list(output[0])

                predicted.append(out_list)

                # compute loss
                err += self.loss(y_train[j], output, x_train[j])

                # backward propagation
                error = self.loss_derivative(y_train[j], output, x_train[j])  # dE/dY
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

                pred = identity_decoder(output[0])
                label = identity_decoder(y_train[j])

                pred = output[0]
                label = y_train[j]

                # print(np.array(label))
                # print(np.array(label)[0])
                # print(1 - np.array(label))
                # print(np.array(pred))
                # print(1 - np.array(pred))
                # print(np.array(label).shape)
                # print(np.array(pred).shape)
                # print(np.array(label).size)

                # calculate accuracy for each sample
                accuracy = float((np.dot(np.array(label), np.array(pred).T) +
                                  np.dot(1 - np.array(label), 1 - np.array(pred).T)) / float(
                    np.array(label).size) * 100)

            # calculate average error and average accuracy on all samples
            # pred = identity_decoder(predicted)
            # label = identity_decoder(y_train)

            # print(np.array(label))
            # print(np.array(label)[0])
            # print(1 - np.array(label))
            # print(np.array(pred))
            # print(1 - np.array(pred))
            # print(np.array(label)[0].shape)
            # print(np.array(pred)[0].shape)
            # print(np.array(label).size)

            # accuracy = float((np.dot(np.array(label)[0], np.array(pred)[0].T) +
            #                   np.dot(1-np.array(label)[0], 1-np.array(pred)[0].T)) / float(np.array(label)[0].size)*100)

            # accuracy /= samples
            err /= samples

            self.error_per_epoch.append(err)
            self.accuracy_per_epoch.append(accuracy)

            print('error={:.9f}         accuracy={:.9f}         for epoch {} of {}'.format(err, accuracy, i+1, epochs))
            # visualize([k for k in range(i+1)], self.error_per_epoch)

        return self.error_per_epoch, self.accuracy_per_epoch

    # predict output for given input
    def predict_output(self, input_data):
        samples = len(input_data)
        out_list, result = [], []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
                out_list = list(output[0])
            result.append(out_list)

        return np.array(result)



