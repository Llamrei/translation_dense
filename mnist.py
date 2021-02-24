"""
Uses code from https://www.tensorflow.org/datasets/keras_example
"""

import tensorflow as tf
import tensorflow_datasets as tfds

from datetime import timedelta, datetime

from trans_dense import tDense


(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

start = datetime.now()

trad_model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128,activation='relu'),
  tf.keras.layers.Dense(10)
])

trad_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

trad_model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
    verbose=0
)
timing = dict()
timing['trad'] = datetime.now() - start
timing['trad'] = timing['trad'] / timedelta(microseconds=1)

res = dict()

res['trad'] = trad_model.evaluate(ds_test, verbose=0)

start = datetime.now()
trans_model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tDense(130,activation='relu'),
  tf.keras.layers.Dense(10, use_bias=False)
])

trans_model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
)

trans_model.fit(
    ds_train,
    epochs=6,
    validation_data=ds_test,
    verbose=0
)
timing['trans'] = datetime.now() - start
timing['trans'] = timing['trans'] / timedelta(microseconds=1)

res['trans'] = trans_model.evaluate(ds_test, verbose=0)