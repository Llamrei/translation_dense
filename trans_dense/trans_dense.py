import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.framework import dtypes
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.framework import tensor_shape

class tDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        input_len = tensor_shape.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_len, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.bias = self.add_weight(
            'bias',
            shape=[input_len,],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs):
        outputs = tf.math.add(inputs, self.bias)
        outputs = tf.linalg.matmul(outputs, self.kernel)
        outputs = self.activation(outputs)
        return outputs