Video Inpainting using 3D CNN
-----------------------------

OS: Ubuntu 16.04 64-bit with NVIDIA GPU support and display monitor

Programming Language: Python 2.7

Python Libraries used:

numpy
matplotlib
pygame
tensorflow-gpu
keras
Pillow
opencv-python
h5py


Project running details:

1. Flickr30k-images Dataset download:

   Please proceed to link:		http://bplumme2.web.engr.illinois.edu/Flickr30kEntities/
   Click the "this form" link provided under "Dataset" in the webpage to access dataset.
   Provide the details and submit your form. You'll receive a link in email to download
   the flickr dataset. Download "Flickr 30k images" and unzip the dataset to the project
   folder with the name "flickr30k-images". Alternatively, you can download other dataset
   but the folder name should be changed in 'create_dataset.py' python file.


2. Download the python libraries provided in requirements.txt file by using the below
   command.

   $ pip2 install --user -r requirements.txt

   Note: Keras uses Tensorflow-gpu backend


3. Check if you've provided correct folder name for flickr dataset in 'create_dataset.py'
   and run the program 'create_dataset.py'. Change parameters as required.
   Default parameters: 50 images -- 50 video_frames -- 50 mini-batches

4. To run the training and testing of the neural network provided in our project at a
   single time - run the bash script using commands provided below

   $ chmod +x runproject.sh
   $ ./runproject.sh

   Alternatively to run the project individually, access the python files 'train.py' and 
   'test.py', change parameters as required and run them.