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

from keras.initializers import glorot_uniform

class NeuralNetwork(object): 
    def __init__(self, data, method, loss, epochs, batch_size, activation='relu', layers=4, kernel_init=glorot_uniform, dropout=0.5, early_stop=125, lr_patience=40, reduce_lr=0.5, reduce_lr_min=0.000009,**kwargs):
        self.data = data
        self.method = method
        self.loss = loss
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
        