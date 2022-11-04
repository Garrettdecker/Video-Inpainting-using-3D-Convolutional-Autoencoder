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

import matplotlib.pyplot as plt
import numpy as np

class disp_fig(object):

    """
    Class to display video frames

    Args:
    Accepts video frames numpy array to display using matplotlib

    Output:
    Displays 8 video frames using matplotlib window
    """

    def __init__(self, arr):
        self.array = arr
        
        # Reshape numpy video array
        self.video = arr.reshape(1,8,152,152)
        
        # Create new matplotlib figure
        plt.figure(figsize=(20, 2))

    def figure(self):
        # Display video frames using 8 subplot
        for i in range(8):
            ax = plt.subplot(1, 8, i+1)
            image = self.video[0][i]
            plt.imshow(image)
            plt.gray()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

if __name__ == '__main__':
	pass