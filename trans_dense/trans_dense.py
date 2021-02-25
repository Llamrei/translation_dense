import tensorflow as tf
from tensorflow.python.framework import tensor_shape

class tDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        print(input_shape)
        print(self.units)
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

class polyDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        print(input_shape)
        print(self.units)
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
            shape=[self.units,input_len],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs):
        outputs = tf.math.add(tf.linalg.matmul(inputs, self.kernel), tf.linalg.diag_part(tf.linalg.matmul(self.bias, self.kernel)) )
        outputs = self.activation(outputs)
        return outputs

class extraDense(tf.keras.layers.Dense):
    def build(self, input_shape):
        print(input_shape)
        print(self.units)
        input_len = tensor_shape.dimension_value(input_shape[-1])
        self.kernel = self.add_weight(
            'kernel',
            shape=[input_len, self.units],
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            dtype=self.dtype,
            trainable=True)
        self.trans = self.add_weight(
            'trans',
            shape=[self.units,input_len],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True
        )
        self.bias = self.add_weight(
            'bias',
            shape=[self.units,],
            initializer=self.bias_initializer,
            regularizer=self.bias_regularizer,
            constraint=self.bias_constraint,
            dtype=self.dtype,
            trainable=True)
        self.built = True

    def call(self, inputs):
        outputs = tf.math.add(tf.linalg.matmul(inputs, self.kernel), tf.linalg.diag_part(tf.linalg.matmul(self.trans, self.kernel)) )
        outputs = tf.nn.bias_add(outputs,self.bias)
        outputs = self.activation(outputs)
        return outputs