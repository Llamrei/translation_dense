"""
Uses code from https://www.tensorflow.org/datasets/keras_example
"""

from trans_dense.trans_dense import extraDense
import tensorflow as tf
import tensorflow_datasets as tfds

import csv
from datetime import timedelta, datetime
import io
import os
import sys
from pathlib import Path

from trans_dense import tDense
from trans_dense import polyDense
from trans_dense import Timer

try:
    OUTPUT_DIR = Path(sys.argv[1])
except IndexError:
    OUTPUT_DIR = Path('.')

VERBOSITY = 0

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

dense_types = {
    'trad' : (tf.keras.layers.Dense, True),
    'trans_wout_bias'  : (tDense, False),
    'trans_w_bias' : (tDense, True),
    'poly_dense_w_bias' : (polyDense, True),
    'poly_dense_wout_bias' : (polyDense, False),
    'extra_dense_w_bias' : (extraDense, True),
    'extra_dense_wout_bias' : (extraDense, False)
}
results = []
for name, (dense, use_bias) in dense_types.items():
    print(f"Training with {name}")
    model = None
    res = dict()
    res['dense_type'] = name
    with Timer() as t:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            dense(128,activation='relu'),
            tf.keras.layers.Dense(10, use_bias=use_bias)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(0.001),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
        )

        t.start("training")
        model.fit(
            ds_train,
            epochs=6,
            validation_data=ds_test,
            verbose=VERBOSITY
        )
    res = {**res, **t.timing}
    res['val_loss'], res['val_acc'] = model.evaluate(ds_test, verbose=VERBOSITY)
    results.append(res)
print(results)

OUTPUT_DIR = OUTPUT_DIR/'mnist'
os.makedirs(OUTPUT_DIR, exist_ok=True)
path = Path(OUTPUT_DIR/'results.txt')
if not path.exists():
    # Use of io open to ensure atomic writing
    with io.open(OUTPUT_DIR/'results.txt', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()

with io.open(path, 'a') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    for res in results:
        writer.writerow(res)