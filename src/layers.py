from keras import backend as K
from keras.engine.topology import Layer

class DensityCount(Layer):

    def call(self, x):
        return K.sum(x, axis=(1,2,3))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
