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

# Program to create 50 video dataset
# Change number to crate more videos

from dataset.resize import resize
from dataset.video import video
from dataset.create_bw_pickle import create_bw_pickle

# Original images folder
flickr_folder = './flickr30k-images/'

# Resized images folder
resized_flickr_folder = './resized_flickr30k-images/'

# Video folder
video_folder = './video-images/'

# Pickle folder
features = './X_train/'
labels = './y_train/'

# Instance
img_data = resize(src_folder=flickr_folder,
                  dest_folder=resized_flickr_folder,
                  num_imgs=50) # Total images = 31784

# Resize images
img_data.resize_images()

# Generate video images
# Instance
video_data = video(src_folder=resized_flickr_folder,
                   dest_folder=video_folder,
                   num_vids=50)

# Create video
video_data.create_videos()

# Create minibatch file
# Create 50 minibatches, each minibatch containing 1 video
create_bw_pickle(feature_src_folder=video_folder,
                 feature_dest_folder=features,
                 label_src_folder=resized_flickr_folder,
                 label_dest_folder=labels,
                 num_vids_in_batch=1,
                 num_of_minibatches=50)
