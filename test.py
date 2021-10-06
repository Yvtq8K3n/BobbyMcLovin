import pathlib

import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt

from IPython.display import Audio

data_dir = pathlib.Path('recordings')
#sample_file = data_dir/'A.mp3'
sample_file = data_dir/'praisethelod_1633289150531_marcode3#8575.mp3' #dtype=float32
audio = tfio.audio.AudioIOTensor(str(sample_file))

# remove last dimension
audio_tensor = tf.cast(audio.to_tensor(), tf.float32) / 32768.0
audio_tensor = tf.math.reduce_mean(audio_tensor, axis=-1)

print("Audio Tensor: " + str(audio_tensor))

plt.figure()
plt.plot(audio_tensor.numpy())
plt.show()

# Convert to spectrogram
spectrogram = tfio.audio.spectrogram(audio_tensor, nfft=512, window=512, stride=256)

plt.figure()
plt.imshow(tf.math.log(spectrogram).numpy())
plt.show()



