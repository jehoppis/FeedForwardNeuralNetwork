import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import FeedForwardNN
from numba import vectorize, cuda

np.random.seed(1)
print('One dimensional curve fitting:')

t = np.arange(-2, 2, 0.04)
print('Number of training points: '+str(t.shape[0]))
t.shape = (t.shape[0], 1)
t2 = np.arange(-2, 2, 0.001)
t2.shape = (t2.shape[0], 1)
print('Training Data shape', t.shape)
print('Test Data shape', t2.shape)


def f(x):
    return x*(x-1)*(x+1)


my_NN = FeedForwardNN.FFNN(1, 1, [25 for i in range(5)])

noise_mag = .5
noise = noise_mag * np.random.random(t.shape)

y1 = f(t)
# y1 = y1 + noise

y2 = my_NN.evaluate(t2)

plt.figure(1)
plt.xlabel('x- axis')
plt.ylabel('Pre-Training NN')
plt.plot(t, y1, 'ro', t2, y2, 'g')

# epochs_, history_ = my_NN.train(t, y1, 0.001, 200)

per = .05
epochs_, history_ = my_NN.train_mini_batches(t, y1, 0.001, 200, .05)

y3 = f(t2)
y4 = my_NN.evaluate(t2)
plt.figure(2)
plt.xlabel('x- axis')
plt.ylabel('Post-Training NN')
plt.plot(t2, y3, 'r', t2, y4, 'g')

plt.figure(3)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.plot(epochs_, history_, 'b')

print('MSE on Test Set:', my_NN.mse(t2, y4))
print()

plt.show()

# Looking under the hood------------------------------------------------------

# print(my_NN.n_hidden_layers)
# print()
# print('Weights:')
# for i in range(0, my_NN.n_hidden_layers + 1):
#     print('w'+str(i), my_NN.weights[i].shape)
# print()
#
# affine_, activation_ = my_NN.forward_prop(t, y1)
# for i in my_NN.weights.keys():
#     print('affine_'+str(i), affine_[i].shape)
#     print('activation_' + str(i), activation_[i].shape)
# print()
#
# error_ = my_NN.back_prop(y1, activation_)
# print('Error')
# for i in my_NN.weights.keys():
#     print('e'+str(i), error_[i].shape)
# print()
#
# partials_ = my_NN.partial_der(t, activation_, error_)
# for i in my_NN.weights.keys():
#     print('partials'+str(i), partials_[i].shape)
# print()

# Surface fitting g: R^2 -> R-----------------------------------------------------------------------
print('Two dimensional surface fitting:')


def g(x,y):
    return (x-y)**2 - (1/3) * x**3


NN2 = FeedForwardNN.FFNN(2, 1, [30 for i in range(3)])

X = np.arange(-3, 3, 0.1)
Y = np.arange(-3, 3, 0.1)
X2, Y2 = np.meshgrid(X, Y)
X.shape = (X.shape[0], 1)
Y.shape = (Y.shape[0], 1)

X3 = np.reshape(X2, (X2.shape[0]*X2.shape[1], 1))
Y3 = np.reshape(Y2, (Y2.shape[0]*Y2.shape[1], 1))

train_input = np.hstack((X3, Y3))
print('Training data shape:', train_input.shape)
train_output = g(train_input[:, 0], train_input[:, 1])
train_output.shape = (train_output.shape[0], 1)

z1 = g(X2, Y2)

# epochs2, history2 = NN2.train(train_input, train_output, .001, 80)

epochs2, history2 = NN2.train_mini_batches(train_input, train_output, 0.001, 80, .01)
z2 = NN2.evaluate(train_input)
z3 = np.reshape(z2, X2.shape)

fig4 = plt.figure(4)
ax4 = fig4.gca(projection='3d')
surf4 = ax4.plot_surface(X2, Y2, z1)
plt.xlabel('x- axis')
plt.ylabel('y-axis')
ax4.set_zlabel('Training Data')
plt.draw()

fig5 = plt.figure(5)
ax5 = fig5.gca(projection='3d')
surf5 = ax5.plot_surface(X2, Y2, z3)
plt.xlabel('x- axis')
plt.ylabel('y-axis')
ax5.set_zlabel('Post-Training NN')
plt.draw()

fig6 = plt.figure(6)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.plot(epochs2, history2, 'b')
plt.draw()

X = np.arange(-3, 3, 0.01)
Y = np.arange(-3, 3, 0.01)
X2, Y2 = np.meshgrid(X, Y)
X.shape = (X.shape[0], 1)
Y.shape = (Y.shape[0], 1)

X3 = np.reshape(X2, (X2.shape[0]*X2.shape[1], 1))
Y3 = np.reshape(Y2, (Y2.shape[0]*Y2.shape[1], 1))

test_data = np.hstack((X3, Y3))
print('Test Data shape', test_data.shape)

z3 = g(test_data[:, 0], test_data[:, 1])
z3 = np.reshape(z3, (X2.shape[0]*X2.shape[1], 1))

print('MSE on Test Set:', NN2.mse(test_data, z3))


plt.show()