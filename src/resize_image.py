#!/usr/bin/python
import os
from PIL import Image

#path = "D:/GitHub/machine_Learning_5JV/images/Test/Tomato"  # Assurez-vous que le chemin est correct

def resize_aspect_fit():
    dirs = os.listdir(path)
    for item in dirs:
        if item.endswith('.jpg'):
            full_path = os.path.join(path, item)
            if os.path.isfile(full_path):
                image = Image.open(full_path)

                max_dimension = max(image.size)
                resize_ratio = 32.0 / max_dimension

                new_image_width = int(image.size[0] * resize_ratio)
                new_image_height = int(image.size[1] * resize_ratio)

                image = image.resize((new_image_width, new_image_height), Image.Resampling.LANCZOS)

                image.save(full_path, 'JPEG', quality=90)

resize_aspect_fit()
