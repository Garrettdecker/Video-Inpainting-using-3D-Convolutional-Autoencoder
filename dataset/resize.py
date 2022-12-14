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

from PIL import Image
import os

SCREEN_LENGTH = 152
SCREEN_WIDTH = 152


class resize(object):
    """
    Class to resize images

    Args:
    src_folder, dest_folder: String variables to accept folder containing images and
    folder to save the resized images

    Output:
    Saves resized images in dest_folder

    """

    def __init__(self, src_folder, dest_folder, num_imgs=1500):
        self.src_dir = src_folder
        self.dest_dir = dest_folder
        self.num_of_imgs = num_imgs
        
        # Create destination folder if it is absent
        if not os.path.exists(self.dest_dir):
            os.makedirs(self.dest_dir)

    def resize_images(self):
        
        # Resize and save images
        image_files = os.listdir(self.src_dir)
        num_images = len(image_files)
        for i, image_file in enumerate(image_files):
            with open(os.path.join(self.src_dir, image_file), 'r+b') as img:
                try:
                    with Image.open(img) as image:
                        
                        # Function call to resize image
                        image = self.resize_image(image)

                        # Resize file name
                        resize_filename = str(i) + '.jpg'
                        
                        # Save the image in dest_folder with with provided name
                        image.save(os.path.join(self.dest_dir, resize_filename),
                                   image.format)
                except IOError:
                    
                    # Continue if it is not a image folder
                    continue

            if i % 100 == 0 and i != 0:
                # Print resize status
                print 'Resized images: %d/%d' % (i, num_images)
            
            if i == self.num_of_imgs:
                # Print number of images resized
                print 'Resized %d images\n' % i
                break

    def resize_image(self, image):
        # Resize image based on below configurations
        width, height = image.size
        if width > height:
            left = (width - height) / 2
            right = width - left
            top = 0
            bottom = height
        else:
            top = (height - width) / 2
            bottom = height - top
            left = 0
            right = width
        image = image.crop((left, top, right, bottom))
        image = image.resize([SCREEN_LENGTH, SCREEN_WIDTH], Image.ANTIALIAS)
        return image

if __name__ == '__main__':
	pass