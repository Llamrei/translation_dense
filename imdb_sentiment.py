"""
Uses code from https://www.tensorflow.org/datasets/keras_example
"""

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

import csv
from datetime import timedelta, datetime
import io
import os
import sys
from pathlib import Path

from trans_dense import tDense
from trans_dense import Timer

try:
    OUTPUT_DIR = Path(sys.argv[1])
except IndexError:
    OUTPUT_DIR = Path('.')

VERBOSITY = 0

# Split the training set into 60% and 40%, so we'll end up with 15,000 examples
# for training, 10,000 examples for validation and 25,000 examples for testing.
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews", 
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True)

embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"

dense_types = {
    'trad' : (tf.keras.layers.Dense, True),
    'trans_wout_bias'  : (tDense, False),
    'trans_w_bias' : (tDense, True)
}
results = []
for name, (dense, use_bias) in dense_types.items():
    print(f"Training with {name}")
    model = None
    res = dict()
    res['dense_type'] = name
    with Timer() as t:
        hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
        model = tf.keras.Sequential([
            hub_layer,
            dense(16, activation='relu'),
            tf.keras.layers.Dense(1, use_bias=use_bias)
        ])
        model.compile(optimizer='adam',
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'])
        t.start("training")
        model.fit(train_data.shuffle(10000).batch(512),
            epochs=10,
            validation_data=validation_data.batch(512),
            verbose=VERBOSITY
            )
    res = {**res, **t.timing}
    res['val_loss'], res['val_acc'] = model.evaluate(validation_data.batch(512), verbose=VERBOSITY)
    results.append(res)
print(results)

OUTPUT_DIR = OUTPUT_DIR/'imdb'
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