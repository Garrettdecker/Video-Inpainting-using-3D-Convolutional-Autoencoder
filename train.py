"""
MIT License

Copyright (c) 2017 Jajati Keshari Routray and Garret Sterling Decker

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

# Auto encoder training code

import numpy as np
from keras.models import Model
from keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D
import matplotlib.pyplot as plt
import h5py
from dataset.disp_fig import disp_fig
import random

input_vid = Input(shape=(8, 152, 152, 1))

# Encoder
x = Conv3D(5, (3, 3, 3), activation='relu', padding='same')(input_vid)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = Conv3D(5, (3, 3, 3), activation='relu', padding='same')(x)
encoded = MaxPooling3D((2, 2, 2), padding='same')(x)

# Video Encoded

# Decoder
x = Conv3D(5, (3, 3, 3), activation='relu', padding='same')(encoded)
x = UpSampling3D((2, 2, 2))(x)
x = Conv3D(5, (3, 3, 3), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)
decoded = Conv3D(1, (3, 3, 3), activation='sigmoid', padding='same')(x)

autoencoder = Model(input_vid, decoded)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Print autoencoder details
autoencoder.summary()

# Train for 50 epochs
nb_epoch = 50
for e in range(nb_epoch):
    print "\n\n\nEpoch ------- %d \n\n\n" % e
    
    # Get a list of random sequence for 49 videos to train
    # Test on 50th video

    rand_list = random.sample(range(49),49)

    for i in rand_list:
        X_name = './X_train/' + str(i) + '.h5'
        y_name = './y_train/' + str(i) + '.h5'
        
        with h5py.File(X_name, 'r') as hf:
            X_train = hf[str(i)][:]

        with h5py.File(y_name, 'r') as hf:
            y_train = hf[str(i)][:]

        # Run autoencoder
        autoencoder.fit(X_train, 
        	            y_train,
        	            verbose=1, 
        	            epochs=1,
        	            batch_size=2, 
        	            shuffle=False,
        	            callbacks=None)


# Save model weights
autoencoder.save_weights('autoencoder_weights.h5')