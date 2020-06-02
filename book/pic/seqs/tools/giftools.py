"""
https://image.online-convert.com/convert/gif-to-png
"""

import os
import numpy as np
import imageio
from PIL import Image

def pngtogif(file_dir = '.',gif_file='movie.gif',fps=50):
    images = []
    for file_name in os.listdir(file_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(file_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave( os.path.join(file_dir, gif_file), images,fps=fps)




def gif2png(file_dir = '.',png_file_root='movie',removeAlpha= False):
    for file_name in os.listdir(file_dir):
        images = []
        if file_name.endswith('.gif'):
            gif = imageio.get_reader(file_name)

            for fcnt,frame in enumerate(gif):
                if removeAlpha:
                    frame[:,:,0:3] = 0  #frame[:,:,0:3]
                # make an image
                img = Image.fromarray(frame)
                imageio.imwrite(uri=f'{png_file_root}-{fcnt:03d}.png', im=img,
                    format='PNG-PIL')

# print(imageio.help('png'))

gif2png(file_dir = '.',png_file_root='movie',removeAlpha=True)

# pngtogif('.','movie.gif',fps)
