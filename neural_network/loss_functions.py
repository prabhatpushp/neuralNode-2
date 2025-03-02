# loss_functions.py

class MeanSquaredError:
    def loss(self, predicted, actual):
        return ( predicted - actual) ** 2

    def gradient(self, predicted, actual):
        return 2 * (predicted - actual)
