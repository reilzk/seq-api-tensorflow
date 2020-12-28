#!/usr/bin/env python 
# Iterasi[1]:


try:
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass

import tensorflow as tf
from tensorflow.python.keras.utils.vis_utils import plot_model
import pydot
from tensorflow.keras.models import Model

# Iterasi ke-[2]:


def build_model_with_sequential():
seq_model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
tf.keras.layers.Dense(128, activation=tf.nn.relu),
tf.keras.layers.Dense(10, activation=tf.nn.softmax)])
    return seq_model

# Iterasi ke-[3]:
#reilzk training API with KERAS [provided from Coursera]

def build_model_with_functional():
    input_layer = tf.keras.Input(shape=(28, 28))
    flatten_layer = tf.keras.layers.Flatten()(input_layer)
    first_dense = tf.keras.layers.Dense(128, activation=tf.nn.relu)(flatten_layer)
    output_layer = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(first_dense)
    func_model = Model(inputs=input_layer, outputs=output_layer)
    
    return func_model

# Iterasi ke-[4]:


model = build_model_with_functional()

# Model grafik
plot_model(model, show_shapes=True, show_layer_names=True, to_file='model.png')

# Iterasi ke-[5]:

#MNIS Dataset
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

#model untuk training
model.compile(optimizer=tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
