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
from keras import Sequential
from keras.api.initializers import glorot_uniform
from keras.api.optimizers import Adam
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Dense, ELU, Activation, Dropout


class Data:
    def __init__(self, data):
        self.data = data
        self.x_train = None
        self.y_train = None
        self.x_valid = None
        self.y_valid = None
        self.x_test = None

    def split_data(self):
        self.x_train, self.x_valid, self.x_test = self.data['x_train'], self.data['x_valid'], self.data['x_test']
        self.y_train, self.y_valid = self.data['y_train'], self.data['y_valid']

    def get_train_data(self):
        return self.x_train, self.y_train

    def get_valid_data(self):
        return self.x_valid, self.y_valid

    def get_test_data(self):
        return self.x_test


class NeuralNetwork(object):
    def __init__(self, data, method, loss='mean_squared_error', neurons=200, epochs=0, batch_size=16, activation='softplus', layers=4,
                 kernel_init=glorot_uniform, dropout=0.5, early_stop=125, lr_patience=40, reduce_lr=0.5,
                 reduce_lr_min=0.000009, **kwargs):
        """
        :param data (Data):  Data class containing the training, validation and testing data
        :param method (keras.api.optimizers):  Optimizer to use for training
        :param loss (str):  Loss function to use for training
        :parm  neurons(int): Number of neurons in the hidden layers
        :param epochs (int):  Number of epochs to train the model
        :param batch_size (int):  Batch size for training
        :param activation (str):  Activation function to use for hidden layers
        :param layers (int):  Number of hidden layers in the model
        :param kernel_init (keras.api.initializers):  Kernel initializer to use for hidden layers
        :param dropout (float):  Dropout percentage to use for hidden layers
        :param early_stop (int): Number of epochs to wait before stopping training if no improvement is made
        :param lr_patience (int):  Number of epochs to wait before reducing learning rate
        :param reduce_lr (float): Factor to reduce learning rate by if no improvement is made
        :param reduce_lr_min (float): Minimum learning rate
        :param kwargs (dict):  Additional keyword arguments
        """
        self.data = Data(data)
        self.method = method
        self.loss = loss
        self.neurons = neurons
        self.epochs = epochs
        self.batch_size = batch_size
        self.activation = activation
        self.layers = layers
        self.kernel_init = kernel_init
        self.dropout = dropout
        self.early_stop = early_stop
        self.lr_patience = lr_patience
        self.reduce_lr = reduce_lr
        self.reduce_lr_min = reduce_lr_min
        self.kwargs = kwargs

    def create_model(self):
        """
        Creates the Calibration Neural Network model
        """
        x_train = self.data.get_train_data()[0]
        y_train = self.data.get_train_data()[1]

        model = Sequential()
        for i in range(self.layers):
            if i == 0:
                model.add(Dense(self.neurons, input_shape=(x_train.shape[1],), kernel_initializer=self.kernel_init))
            else:
                model.add(Dense(self.neurons, kernel_initializer=self.kernel_init))
            if self.activation == 'elu':
                model.add(ELU())
            else:
                model.add(Activation(self.activation))
            model.add(Dropout(self.dropout))
            model.add(Dense(y_train.shape[1], kernel_initializer=self.kernel_init))
            model.compile(self.method, loss=self.loss)

        return model

    def train_model(self):
        """
        Trains the model
        """
        x_train, y_train = self.data.get_train_data()
        x_valid, y_valid = self.data.get_valid_data()
        x_test = self.data.get_test_data()
        model = self.create_model()
        if self.epochs > 0:
            callbacks = []
            if self.early_stop is not None:
                early_stopping = EarlyStopping(monitor='val_loss', patience=self.early_stop)
                callbacks.append(early_stopping)
            if self.reduce_lr is not None:
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=self.reduce_lr, patience=self.lr_patience,
                                              min_lr=self.reduce_lr_min, verbose=1)
                callbacks.append(reduce_lr)
            history = model.fit(x_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                                validation_data=(x_valid, y_valid), callbacks=callbacks)
        else:
            history = None

        return x_train, x_valid, x_test, model, history