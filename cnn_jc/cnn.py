# Neural Network Parameters
# epochs: Number of epochs to train the model
# residual_cells: Sets the number of layers that will be skipped in building residual feedback
# learning_rate: Learning rate for the optimizer
# layers: Number of hidden layers in the model (default: 4)
# dropout: Percentage of cells randomly dropped in training (default: 0.5)
# earlyStopPatience: Number of epochs to wait before stopping training if no improvement is made
# reduceLRFactor: Factor to reduce learning rate by if no improvement is made
# reduceLRPatience: Number of epochs to wait before reducing learning rate
# reduceLRMin: Minimum learning rate

# class Data:
#     def __init__(self, train_data, test_data):
#         self.x_train = train_data['x_train']
#         self.y_train = train_data['y_train']
#         self.x_test = test_data['x_test']
#
#     def get_train_data(self):
#         return self.x_train, self.y_train
#
#     def get_test_data(self):
#         return self.x_test
#
#
# class NeuralNetwork(object):
#     def __init__(self, data, method, loss='mean_squared_error', neurons=200, epochs=0, batch_size=16,
#                  activation='softplus', layers=4,
#                  kernel_init=glorot_uniform, dropout=0.5, early_stop=125, lr_patience=40, reduce_lr=0.5,
#                  reduce_lr_min=0.000009, **kwargs):
#         """
#         :param data (Data):  Data class containing the training, validation and testing data
#         :param method (keras.api.optimizers):  Optimizer to use for training
#         :param loss (str):  Loss function to use for training
#         :parm  neurons(int): Number of neurons in the hidden layers
#         :param epochs (int):  Number of epochs to train the model
#         :param batch_size (int):  Batch size for training
#         :param activation (str):  Activation function to use for hidden layers
#         :param layers (int):  Number of hidden layers in the model
#         :param kernel_init (keras.api.initializers):  Kernel initializer to use for hidden layers
#         :param dropout (float):  Dropout percentage to use for hidden layers
#         :param early_stop (int): Number of epochs to wait before stopping training if no improvement is made
#         :param lr_patience (int):  Number of epochs to wait before reducing learning rate
#         :param reduce_lr (float): Factor to reduce learning rate by if no improvement is made
#         :param reduce_lr_min (float): Minimum learning rate
#         :param kwargs (dict):  Additional keyword arguments
#         """
#         self.data = Data(data)
#         self.method = method
#         self.loss = loss
#         self.neurons = neurons
#         self.epochs = epochs
#         self.batch_size = batch_size
#         self.activation = activation
#         self.layers = layers
#         self.kernel_init = kernel_init
#         self.dropout = dropout
#         self.early_stop = early_stop
#         self.lr_patience = lr_patience
#         self.reduce_lr = reduce_lr
#         self.reduce_lr_min = reduce_lr_min
#         self.kwargs = kwargs
#
#     def create_model(self):
#         """
#         Creates the Calibration Neural Network model
#         """
#         x_train, y_train = self.data.get_train_data()
#
#         model = Sequential()
#         for i in range(self.layers):
#             if i == 0:
#                 model.add(Dense(self.neurons, input_shape=(x_train.shape[1],), kernel_initializer=self.kernel_init))
#             else:
#                 model.add(Dense(self.neurons, kernel_initializer=self.kernel_init))
#             if self.activation == 'elu':
#                 model.add(ELU())
#             else:
#                 model.add(Activation(self.activation))
#             model.add(Dropout(self.dropout))
#             model.add(Dense(y_train.shape[1], kernel_initializer=self.kernel_init))
#             model.compile(self.method, loss=self.loss)
#
#         return model
#
#     def train_model(self):
#         """
#         Trains the model
#         """
#         x_train, y_train = self.data.get_train_data()
#         # x_valid, y_valid = self.data.get_valid_data()
#         x_test = self.data.x_test
#         model = self.create_model()
#         if self.epochs > 0:
#             callbacks = []
#             if self.early_stop is not None:
#                 early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stop)
#                 callbacks.append(early_stopping)
#             if self.reduce_lr is not None:
#                 reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.reduce_lr, patience=self.lr_patience,
#                                               min_lr=self.reduce_lr_min, verbose=1)
#                 callbacks.append(reduce_lr)
#             history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
#                                 callbacks=callbacks)
#         else:
#             history = None
#
#         return x_train, x_test, model, history


from sklearn.neural_network import MLPRegressor
import numpy as np
from matplotlib import pyplot as plt

class Data:
    def __init__(self, train_data, test_data):
        self.x_train = train_data['x_train']
        self.y_train = train_data['y_train']
        self.x_test = test_data['x_test']
        self.y_test = test_data['y_test']

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_test_data(self):
        return self.x_test, self.y_test


class NeuralNetwork(object):
    def __init__(self, data, loss='mean_squared_error', neurons=200, epochs=200, batch_size=16,
                 activation='relu', layers=4, alpha=0.0001, learning_rate_init=0.001,
                 learning_rate='adaptive', early_stopping=True, n_iter_no_change=10, tol=1e-4, **kwargs):
        """
        :param data (Data):  Data class containing the training, validation and testing data
        :param loss (str):  Loss function to use for training
        :param neurons(int): Number of neurons in the hidden layers
        :param epochs (int):  Number of epochs to train the model
        :param batch_size (int):  Batch size for training
        :param activation (str):  Activation function to use for hidden layers
        :param layers (int):  Number of hidden layers in the model
        :param alpha (float): L2 regularization parameter
        :param learning_rate_init (float): Initial learning rate
        :param learning_rate (str): Learning rate schedule to use
        :param early_stopping (bool): Whether to use early stopping
        :param n_iter_no_change (int): Number of epochs to wait before stopping training if no improvement is made
        :param tol (float): Minimum improvement in the validation loss required to continue training
        :param kwargs (dict):  Additional keyword arguments
        """
        self.data = data
        self.loss = loss
        self.neurons = neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.layers = layers
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.learning_rate = learning_rate
        self.early_stopping = early_stopping
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.kwargs = kwargs

    def create_model(self):
        """
        Creates the Calibration Neural Network model
        """
        model = MLPRegressor(hidden_layer_sizes=(self.neurons,) * self.layers, activation=self.activation,
                             alpha=self.alpha, max_iter=self.epochs, batch_size=self.batch_size,
                             learning_rate_init=self.learning_rate_init, learning_rate=self.learning_rate,
                             early_stopping=self.early_stopping, n_iter_no_change=self.n_iter_no_change,
                             tol=self.tol,**self.kwargs)
        return model
    def train_model(self):
        """
        Trains the model
        """
        x_train, y_train = self.data.get_train_data()
        x_test, y_test = self.data.get_test_data()
        
        # Flatten the input data if it has more than 2 dimensions
        if x_train.ndim > 2:
            x_train = np.reshape(x_train, (x_train.shape[0], -1))
        if x_test.ndim > 2:
            x_test = np.reshape(x_test, (x_test.shape[0], -1))
        if y_test.ndim > 2:
            y_test = np.reshape(y_test, (y_test.shape[0], -1))
        
        # Flatten the output data if it has more than 2 dimensions
        if y_train.ndim > 2:
            y_train = np.reshape(y_train, (y_train.shape[0], -1))
        
        # Convert y_train to a 1-dimensional array
        y_train = np.ravel(y_train)
        
        model = self.create_model()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, c='blue', marker='o', label='Test Data')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Calibration Plot')
        plt.show()
        
        
        return x_train, x_test, model