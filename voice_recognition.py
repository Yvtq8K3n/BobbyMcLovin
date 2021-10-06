import os
import pathlib

import datetime
import pickle
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_io as tfio

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def get_waveform_and_label(file_path):
  #Label
  path = tf.strings.split(file_path, os.path.sep)
  label = path[-2]

  #Decode Audio
  audio_binary = tf.io.read_file(file_path)
  audio, _ = tf.audio.decode_wav(audio_binary)
  waveform = tf.squeeze(audio, axis=-1)

  return waveform, label

def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  return spectrogram

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE) 
  output_ds = output_ds.map(get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  print(output_ds)
  return output_ds #return tuple (spetogram, label)

print("Loading model")
model = tf.keras.models.load_model('models/model_0.8_2021-10-04.h5')
CLASS_NAMES = pickle.loads(open("models/labels_0.8_2021-10-04.pickle", "rb").read())

data_dir = pathlib.Path('recordings')
if not data_dir.exists():
    print("Unable to load recordings")

AUTOTUNE = tf.data.AUTOTUNE
#sample_file = data_dir/'kekeres/testB.wav'
sample_file = data_dir/'kekeres/kekeres_1633285520639_SWarrior#3363.wav'
sample_ds = preprocess_dataset([str(sample_file)])


for spectrogram in sample_ds.batch(1):
  prediction = model(spectrogram)
  prediction_data = tf.nn.softmax(prediction, axis=1).numpy()

  index = tf.argmax(prediction, axis=1) 
  print("class: "+str(CLASS_NAMES[np.array(index)]))
  print("confidence: "+str(prediction_data[0][np.array(index)]))
  
  

    #data: tf.Tensor([0.34397040 0.0010119698e 0.00077650481e 0.96381456e] -> yes
    #daata: tf.Tensor([9.9998391e-01 7.9008116e-09 1.4693341e-10 1.6124874e-05]  ->  KEKERRES