import numpy as np

class FFNN:
    def __init__(self, input_size, output_size, hidden_layers):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.n_hidden_layers = len(self.hidden_layers)
        self.weights = dict()
        self.weights[0] = np.vstack((np.random.randn(self.input_size, self.hidden_layers[0]) * np.sqrt(2 / self.hidden_layers[0]),
                           np.zeros((1, self.hidden_layers[0]))))
        for i in range(1, self.n_hidden_layers):
            self.weights[i] = np.vstack((np.random.randn(self.hidden_layers[i-1], self.hidden_layers[i]) *
                               np.sqrt(2 / self.hidden_layers[i]), np.zeros((1, self.hidden_layers[i]))))
        self.weights[self.n_hidden_layers] = np.vstack((np.random.randn(self.hidden_layers[self.n_hidden_layers - 1],
                                                              self.output_size) * np.sqrt(2/ output_size),
                                              np.zeros((1, self.output_size))))

    def relu(self, x):
        return x * (x > 0)
        # return x

    def reluD(self, x):
        return np.array(x > 0, dtype=float)
        # return np.ones(x.shape)

    def permute(self, input_data, output_data):
        A = np.hstack((input_data, output_data))
        np.random.shuffle(A)
        new_input = A[:, 0: input_data.shape[1]]
        new_output = A[:, input_data.shape[1]:]
        return new_input, new_output

    def evaluate(self, input_data):
        output_data = np.hstack((input_data, np.ones((input_data.shape[0], 1))))
        for i in self.weights.keys():
            if i != self.n_hidden_layers:
                output_data = self.relu(np.matmul(output_data, self.weights[i]))
                output_data = np.hstack((output_data, np.ones((output_data.shape[0], 1))))
            else:
                output_data = np.matmul(output_data, self.weights[i])
        return output_data

    def mse(self, input_data, output_data):
        return (1/(2*input_data.shape[0])) * np.sum(np.square(self.evaluate(input_data) - output_data))

    def forward_prop(self, input_data, output_data):
        affine = dict()
        activation = dict()
        for i in range(self.n_hidden_layers + 1):
            if i == 0:
                affine[0] = np.matmul(np.hstack((input_data, np.ones((input_data.shape[0], 1)))), self.weights[0])
                activation[0] = np.hstack((self.relu(affine[0]), np.ones((input_data.shape[0], 1))))
            elif i < self.n_hidden_layers:
                affine[i] = np.matmul(activation[i - 1], self.weights[i])
                activation[i] = np.hstack((self.relu(affine[i]), np.ones((activation[i - 1].shape[0], 1))))
            else:
                affine[i] = np.matmul(activation[i - 1], self.weights[i])
                activation[i] = affine[i]
        return affine, activation

    def back_prop(self, output_data, activation):
        error = dict()
        activation_no_bias = dict()
        for i in range(self.n_hidden_layers, -1, -1):
            activation_no_bias[i] = activation[i][:, 0:activation[i].shape[1] - 1]
            if i == self.n_hidden_layers:
                error[self.n_hidden_layers] = activation[self.n_hidden_layers] - output_data
            else:
                error[i] = self.reluD(activation_no_bias[i]) * np.matmul(error[i+1],
                                                                         self.weights[i+1].T[:,
                                                                         0:self.weights[i+1].shape[0] - 1])
        return error

    def partial_der(self, input_data, activation, error):
        partials = dict()
        for i in range(self.n_hidden_layers, -1, -1):
            if i != 0:
                partials[i] = np.matmul(activation[i - 1].T, error[i])
            else:
                partials[0] = np.matmul(np.hstack((input_data, np.ones((input_data.shape[0], 1)))).T, error[0])
        return partials

    def update(self, input_data, partials, learning_rate):
        for i in self.weights.keys():
            self.weights[i] += -(learning_rate / input_data.shape[0]) * partials[i]

    def train(self, input_data, output_data, learning_rate, n_epochs):
        loss_list = []
        input_data, output_data = self.permute(input_data, output_data)
        for i in range(n_epochs):
            affine, activation = self.forward_prop(input_data, output_data)
            error = self.back_prop(output_data, activation)
            partials = self.partial_der(input_data, activation, error)
            self.update(input_data, partials, learning_rate)
            loss_list.append(self.mse(input_data, output_data))
        epochs = np.arange(0, n_epochs, 1)
        history = np.array(loss_list)
        history.shape = epochs.shape
        return epochs, history

    def train_mini_batches(self, input_data, output_data, learning_rate, n_epochs, data_percent):
        loss_list = []
        input_data, output_data = self.permute(input_data, output_data)
        batch_size = int(data_percent * input_data.shape[0])
        n_batches = input_data.shape[0] // batch_size
        
        for i in range(n_epochs):
            for j in range(n_batches):
                k = j * batch_size
                affine, activation = self.forward_prop(input_data[k:k+batch_size, :], output_data[k:k+batch_size, :])
                error = self.back_prop(output_data[k :k+batch_size, :], activation)
                partials = self.partial_der(input_data[k:k+batch_size, :], activation, error)
                self.update(input_data[k:k+batch_size, :], partials, learning_rate)
                
            if n_batches * batch_size != input_data.shape[0]:
                affine, activation = self.forward_prop(input_data[n_batches * batch_size:, :],
                                                       output_data[n_batches * batch_size:, :])
                error = self.back_prop(output_data[n_batches * batch_size:, :], activation)
                partials = self.partial_der(input_data[n_batches * batch_size:, :], activation, error)
                self.update(input_data[n_batches * batch_size:, :], partials, learning_rate)
                
            loss_list.append(self.mse(input_data, output_data))
        epochs = np.arange(0, n_epochs, 1)
        history = np.array(loss_list)
        history.shape = epochs.shape
        return epochs, history
