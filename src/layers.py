from keras import backend as K
from keras.engine.topology import Layer

class DensityCount(Layer):

    #def __init__(self, output_dim, **kwargs):
    #    self.output_dim = output_dim
    #    super(DensityCount, self).__init__(**kwargs)

    #def build(self, input_shape):
    #    super(DensityCount, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.sum(x, axis=(1,2,3))

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)
